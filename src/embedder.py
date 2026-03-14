"""OptimizedEmbedder – embedding encoding and gradient-based optimization."""

from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from .checkpoint import CheckpointManager


class OptimizedEmbedder:

    def __init__(self, model_name: str, checkpoint_manager: CheckpointManager = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.checkpoint_manager = checkpoint_manager

    def encode(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts, return_tensors="pt",
            padding=True, truncation=True, max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]  # CLS
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def optimize_embedding(
        self, query: str, positives: List[str], negatives: List[str],
        dataset: str, num_steps: int, lr: float,
        reg_weight: float, pos_weight: float, neg_weight: float,
        verbose: bool = False,
        decompose_model: str = None, prompt_version: str = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        # Check cache
        if use_cache and self.checkpoint_manager:
            cached = self.checkpoint_manager.load_optimized_embedding(
                query, dataset, reg_weight, pos_weight, neg_weight,
                num_steps, lr,
                embed_model=self.model_name,
                decompose_model=decompose_model,
                prompt_version=prompt_version,
            )
            if cached:
                emb, metadata = cached
                print(f"  [Cache Hit] {query[:40]}... ({self.model_name})")
                if verbose and "loss_history" in metadata:
                    history = metadata["loss_history"]
                    if "total_losses" in history:
                        print(f"    Cached: initial={history['total_losses'][0]:.4f}, "
                              f"final={history['total_losses'][-1]:.4f}")
                return torch.tensor(emb, device=self.device)

        # Original embedding
        orig_emb = self.encode(query, normalize=True)

        pos_embs = self.encode(positives, normalize=True) if positives else None
        neg_embs = self.encode(negatives, normalize=True) if negatives else None

        updated_emb = orig_emb.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([updated_emb], lr=lr)

        history = {
            "total_losses": [], "pos_losses": [],
            "deviation_losses": [], "neg_losses": [],
        }

        for step in range(num_steps):
            optimizer.zero_grad()

            pos_loss = torch.tensor(0.0, device=self.device)
            if pos_embs is not None and len(pos_embs) > 0:
                pos_loss = torch.norm(updated_emb - pos_embs, dim=1).mean()

            dev_loss = torch.norm(updated_emb - orig_emb)

            neg_loss = torch.tensor(0.0, device=self.device)
            if neg_embs is not None and len(neg_embs) > 0:
                neg_loss = torch.norm(updated_emb - neg_embs, dim=1).mean()

            loss = pos_weight * pos_loss + reg_weight * dev_loss - neg_weight * neg_loss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                updated_emb.data = F.normalize(updated_emb.data, p=2, dim=-1)

            history["total_losses"].append(loss.item())
            history["pos_losses"].append(pos_loss.item())
            history["deviation_losses"].append(dev_loss.item())
            history["neg_losses"].append(neg_loss.item())

            if verbose and step % 100 == 0:
                print(f"    Step {step:3d} | Loss: {loss.item():.4f} | "
                      f"Pos: {pos_loss.item():.4f} | "
                      f"Reg: {dev_loss.item():.4f} | "
                      f"Neg: {neg_loss.item():.4f}")

        final_emb = updated_emb.detach()

        # Save
        if self.checkpoint_manager:
            metadata = {
                "embed_model": self.model_name,
                "num_steps": num_steps, "lr": lr,
                "reg_weight": reg_weight,
                "pos_weight": pos_weight,
                "neg_weight": neg_weight,
                "final_loss": history["total_losses"][-1],
                "initial_loss": history["total_losses"][0],
                "positive_queries": positives,
                "negative_queries": negatives,
                "loss_history": history,
            }
            self.checkpoint_manager.save_optimized_embedding(
                query, final_emb.cpu().numpy(), metadata, dataset,
                decompose_model=decompose_model,
                prompt_version=prompt_version,
            )

        return final_emb
