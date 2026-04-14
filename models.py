from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, TGNMemory


class BaseTemporalModel(nn.Module, ABC):  # abstract class
    """Common interface for temporal node classification models."""

    @abstractmethod  # decorator enforces that inheriting classses have to implement this method
    def reset_state(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def detach_state(
        self,
    ) -> None:  # detaches computation graph (prevents looking back too far in time)
        raise NotImplementedError

    @abstractmethod
    def get_logits(
        self, *, label_nodes: Tensor, current_time: int
    ) -> Tensor:  # raw output predictions for specific ndoes at specific times
        raise NotImplementedError

    @abstractmethod
    def update_event(
        self, *, src: Tensor, dst: Tensor, t: Tensor, msg: Tensor
    ) -> None:  # models memory/state is updates with the new event
        raise NotImplementedError

    @abstractmethod
    def inject_node_features(self, *, node_features: Tensor) -> None:
        """Inject causal node initialization features used by the readout path."""
        raise NotImplementedError


class TGNClassifier(
    BaseTemporalModel
):  # inherics abstract class so we must implement those four functions
    def __init__(
        self,
        *,  # config parameters must be passed by keywords
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        num_classes: int,
        node_feature_dim: int | None,
    ) -> None:
        super().__init__()
        self.memory = TGNMemory(  # TGNMemory module. storage for num_nodes dimensions of messages, memory, and time embeddings.
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(
                raw_msg_dim, memory_dim, time_dim
            ),  # how incoming messages are formatted
            aggregator_module=LastAggregator(),  # grabs the last message recieved
        )
        # we build the memory embedding then pass that to a classifier MLP to make the final predictions
        self.classifier = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, num_classes),
        )
        self.node_feature_proj = (
            nn.Linear(node_feature_dim, memory_dim) if node_feature_dim is not None else None
        )
        if node_feature_dim is not None:
            self.register_buffer("node_features", torch.zeros(num_nodes, node_feature_dim))

    def reset_state(self) -> None:  # clears the memory
        self.memory.reset_state()

    def detach_state(self) -> None:  # detaches the memory
        self.memory.detach()

    def get_logits(self, *, label_nodes: Tensor, current_time: int) -> Tensor:
        # label_nodes: (num_labels,)
        n_id = torch.unique(label_nodes)  # finds the unique nodes -> (num_unique,)
        memory_emb, _ = self.memory(
            n_id
        )  # gets memory embeddings for every unique node -> (num_unique, memory_dim)
        per_label = memory_emb[
            torch.searchsorted(n_id, label_nodes)
        ]  # embeddings mapped back to the label nodes -> (num_labels, memory_dim)
        if self.node_feature_proj is not None:
            node_emb = self.node_feature_proj(self.node_features[n_id])
            per_label = per_label + node_emb[torch.searchsorted(n_id, label_nodes)]
        return self.classifier(
            per_label
        )  # MLP creates final predicitons -> (num_labels, num_classes)

    def update_event(self, *, src: Tensor, dst: Tensor, t: Tensor, msg: Tensor) -> None:
        self.memory.update_state(src=src, dst=dst, t=t, raw_msg=msg)

    def inject_node_features(self, *, node_features: Tensor) -> None:
        if self.node_feature_proj is None:
            return
        self.node_features.copy_(node_features.to(self.node_features.device))


class TGATClassifier(BaseTemporalModel):
    """TGAT-style model built on TGN memory plus temporal self-attention."""

    def __init__(
        self,
        *,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        num_classes: int,
        heads: int,
        node_feature_dim: int | None,
    ) -> None:
        super().__init__()
        # follows same setup as TGN however uses an attention mechanism on previous embeds
        self.memory = TGNMemory(  # same setup as TGN
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(raw_msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        self.time_encoder = nn.Linear(
            1, memory_dim
        )  # turns the delta time into a vector
        self.time_norm = nn.LayerNorm(memory_dim)
        self.attn = nn.MultiheadAttention(  # multihead attention to weight the importance of previous features
            embed_dim=memory_dim,
            num_heads=heads,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(memory_dim)
        self.register_buffer("max_abs_dt", torch.tensor(1_000_000.0))
        self.classifier = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, num_classes),
        )
        self.node_feature_proj = (
            nn.Linear(node_feature_dim, memory_dim) if node_feature_dim is not None else None
        )
        if node_feature_dim is not None:
            self.register_buffer("node_features", torch.zeros(num_nodes, node_feature_dim))

    def reset_state(self) -> None:
        self.memory.reset_state()  # resets the state of the nodes' memories

    def detach_state(self) -> None:
        self.memory.detach()  # removes the nodes' memories from compute graph

    def get_logits(self, *, label_nodes: Tensor, current_time: int) -> Tensor:
        # label_nodes: (num_labels,)
        n_id = torch.unique(label_nodes)  # -> (num_unique,)
        memory_emb, last_update = self.memory(
            n_id
        )  # memory_emb: (num_unique, memory_dim), last_update: (num_unique,)

        # Build a lightweight temporal token by combining memory with delta-time encoding.
        dt = (float(current_time) - last_update).unsqueeze(-1)  # dt: (num_unique, 1)
        dt = torch.clamp(dt, min=-self.max_abs_dt.item(), max=self.max_abs_dt.item())
        time_emb = self.time_norm(self.time_encoder(dt))
        token = memory_emb + time_emb  # token: (num_unique, memory_dim)
        attn_out, _ = self.attn(
            token.unsqueeze(0), token.unsqueeze(0), token.unsqueeze(0)
        )  # attn_out: (1, num_unique, memory_dim)
        enriched = self.attn_norm(attn_out.squeeze(0))  # enriched: (num_unique, memory_dim)
        if self.node_feature_proj is not None:
            enriched = enriched + self.node_feature_proj(self.node_features[n_id])
        # i think the context window here is just the memory of the node, potential for increasing this

        per_label = enriched[
            torch.searchsorted(n_id, label_nodes)
        ]  # per_label: (num_labels, memory_dim)
        return self.classifier(per_label)  # -> (num_labels, num_classes)

    def update_event(self, *, src: Tensor, dst: Tensor, t: Tensor, msg: Tensor) -> None:
        self.memory.update_state(src=src, dst=dst, t=t, raw_msg=msg)

    def inject_node_features(self, *, node_features: Tensor) -> None:
        if self.node_feature_proj is None:
            return
        self.node_features.copy_(node_features.to(self.node_features.device))


class DyRepClassifier(BaseTemporalModel):
    """DyRep-style model with explicit decayed state + recurrent updates."""

    def __init__(
        self,
        *,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        num_classes: int,
        node_feature_dim: int | None,
    ) -> None:
        super().__init__()
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(raw_msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        # same same

        # raw messages projected into the memory dimension
        self.msg_proj = nn.Linear(raw_msg_dim, memory_dim)
        # rnn used to recursively update node states
        self.gru = nn.GRUCell(memory_dim, memory_dim)
        # learnable metric for how much a node's state should decay over time
        self.decay = nn.Parameter(torch.tensor(0.001))
        self.classifier = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, num_classes),
        )

        # non learnable tensors that are used in the computation
        self.register_buffer("node_state", torch.zeros(num_nodes, memory_dim))
        self.register_buffer("last_seen", torch.zeros(num_nodes))
        self.node_feature_proj = (
            nn.Linear(node_feature_dim, memory_dim) if node_feature_dim is not None else None
        )
        if node_feature_dim is not None:
            self.register_buffer("node_features", torch.zeros(num_nodes, node_feature_dim))

    def reset_state(self) -> None:
        self.memory.reset_state()
        self.node_state.zero_()
        self.last_seen.zero_()  # type: ignore

    def detach_state(self) -> None:
        self.memory.detach()
        self.node_state = self.node_state.detach()

    def _decayed_state(self, nodes: Tensor, current_time: int) -> Tensor:
        # helper method to decay node state based on when it was last seen
        dt = float(current_time) - self.last_seen[nodes]  # type: ignore
        # ensure that decay is always positive
        decay = torch.exp(-torch.relu(self.decay) * dt).unsqueeze(-1)
        # updates to the nodes
        return self.node_state[nodes] * decay

    def get_logits(self, *, label_nodes: Tensor, current_time: int) -> Tensor:
        unique_nodes = torch.unique(label_nodes)
        memory_emb, _ = self.memory(unique_nodes)  # get memory embedding
        # we add in the decaying node features
        blended = memory_emb + self._decayed_state(unique_nodes, current_time)
        if self.node_feature_proj is not None:
            blended = blended + self.node_feature_proj(self.node_features[unique_nodes])
        # applying the labels
        per_label = blended[torch.searchsorted(unique_nodes, label_nodes)]
        # classify
        return self.classifier(per_label)

    def update_event(self, *, src: Tensor, dst: Tensor, t: Tensor, msg: Tensor) -> None:
        # update the memory
        self.memory.update_state(src=src, dst=dst, t=t, raw_msg=msg)

        # identifies the nodes that are part of this event batch
        event_t = float(t[0].item())
        event_msg = self.msg_proj(msg[0])
        touched = torch.unique(torch.cat([src, dst], dim=0))

        # for each effected node we get their state
        prev = self.node_state[touched]
        # update their state based on GRU
        new_state = self.gru(event_msg.expand_as(prev), prev)
        # node state and last_seen updated (these are just our memory tensors)
        self.node_state[touched] = new_state
        self.last_seen[touched] = event_t  # type: ignore
        # we predict based on our custom memory tensors as well as the TGN memory module still

    def inject_node_features(self, *, node_features: Tensor) -> None:
        if self.node_feature_proj is None:
            return
        self.node_features.copy_(node_features.to(self.node_features.device))


class EvolveGCNOClassifier(BaseTemporalModel):
    """Lightweight transformer-like temporal model using evolving classifier state."""

    def __init__(
        self,
        *,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        num_classes: int,
        node_feature_dim: int | None,
    ) -> None:
        super().__init__()
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(raw_msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        self.msg_proj = nn.Linear(raw_msg_dim, memory_dim)
        self.controller = nn.GRUCell(memory_dim, memory_dim)
        self.base_classifier = nn.Linear(memory_dim, num_classes)  # makes predictions
        self.delta_classifier = nn.Linear(
            memory_dim, num_classes
        )  # adapts predictions based on global state

        self.register_buffer("global_state", torch.zeros(memory_dim))
        self.node_feature_proj = (
            nn.Linear(node_feature_dim, memory_dim) if node_feature_dim is not None else None
        )
        if node_feature_dim is not None:
            self.register_buffer("node_features", torch.zeros(num_nodes, node_feature_dim))

    def reset_state(self) -> None:
        self.memory.reset_state()
        self.global_state.zero_()

    def detach_state(self) -> None:
        self.memory.detach()
        self.global_state = self.global_state.detach()

    def get_logits(self, *, label_nodes: Tensor, current_time: int) -> Tensor:
        unique_nodes = torch.unique(label_nodes)
        memory_emb, _ = self.memory(unique_nodes)  # same same
        if self.node_feature_proj is not None:
            memory_emb = memory_emb + self.node_feature_proj(self.node_features[unique_nodes])
        dynamic_bias = self.delta_classifier(self.global_state).unsqueeze(
            0
        )  # bias proj based on global state
        logits = self.base_classifier(memory_emb) + dynamic_bias  # biased predictions
        return logits[torch.searchsorted(unique_nodes, label_nodes)]

    def update_event(self, *, src: Tensor, dst: Tensor, t: Tensor, msg: Tensor) -> None:
        self.memory.update_state(
            src=src, dst=dst, t=t, raw_msg=msg
        )  # updates standard memory
        control = self.msg_proj(msg[0])
        self.global_state = self.controller(  # GRU to control global state
            control.unsqueeze(0), self.global_state.unsqueeze(0)
        ).squeeze(0)

    def inject_node_features(self, *, node_features: Tensor) -> None:
        if self.node_feature_proj is None:
            return
        self.node_features.copy_(node_features.to(self.node_features.device))


class GraphSAGEClassifier(BaseTemporalModel):
    """GraphSAGE-style baseline over observed temporal interaction graph."""

    def __init__(
        self,
        *,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        num_classes: int,
        node_feature_dim: int | None,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(raw_msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        self.sage1 = SAGEConv(memory_dim, memory_dim)
        self.sage2 = SAGEConv(memory_dim, memory_dim)
        self.classifier = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, num_classes),
        )

        self.register_buffer("node_state", torch.zeros(num_nodes, memory_dim))
        self._adj: dict[int, set[int]] = {i: set() for i in range(num_nodes)}
        self.node_feature_proj = (
            nn.Linear(node_feature_dim, memory_dim) if node_feature_dim is not None else None
        )
        if node_feature_dim is not None:
            self.register_buffer("node_features", torch.zeros(num_nodes, node_feature_dim))

    def reset_state(self) -> None:
        self.memory.reset_state()
        self.node_state.zero_()
        self._adj = {i: set() for i in range(self.num_nodes)}

    def detach_state(self) -> None:
        self.memory.detach()
        self.node_state = self.node_state.detach()

    def _local_subgraph(self, label_nodes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        label_list = torch.unique(label_nodes).detach().cpu().tolist()
        local_set = set(int(x) for x in label_list)
        for node in list(local_set):
            local_set.update(self._adj.get(node, set()))

        local_nodes = torch.tensor(sorted(local_set), dtype=torch.long, device=label_nodes.device)
        id_map = {int(n.item()): i for i, n in enumerate(local_nodes.detach().cpu())}

        edges_u: list[int] = []
        edges_v: list[int] = []
        for u in local_set:
            for v in self._adj.get(u, set()):
                if v in id_map:
                    edges_u.append(id_map[u])
                    edges_v.append(id_map[v])

        if not edges_u:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=label_nodes.device)
        else:
            edge_index = torch.tensor([edges_u, edges_v], dtype=torch.long, device=label_nodes.device)

        label_pos = torch.searchsorted(local_nodes, label_nodes)
        return local_nodes, edge_index, label_pos

    def get_logits(self, *, label_nodes: Tensor, current_time: int) -> Tensor:
        local_nodes, edge_index, label_pos = self._local_subgraph(label_nodes)
        x = self.node_state[local_nodes]

        if edge_index.numel() > 0:
            x = torch.relu(self.sage1(x, edge_index))
            x = torch.relu(self.sage2(x, edge_index))

        return self.classifier(x[label_pos])

    def update_event(self, *, src: Tensor, dst: Tensor, t: Tensor, msg: Tensor) -> None:
        self.memory.update_state(src=src, dst=dst, t=t, raw_msg=msg)

        touched = torch.unique(torch.cat([src, dst], dim=0))
        emb, _ = self.memory(touched)
        self.node_state[touched] = emb.detach()

        src_i = int(src[0].item())
        dst_i = int(dst[0].item())
        self._adj[src_i].add(dst_i)
        self._adj[dst_i].add(src_i)

    def inject_node_features(self, *, node_features: Tensor) -> None:
        if self.node_feature_proj is None:
            return
        self.node_features.copy_(node_features.to(self.node_features.device))
        injected = self.node_feature_proj(self.node_features)
        self.node_state = 0.5 * self.node_state + 0.5 * injected


def build_model(
    *,
    model_name: str,
    num_nodes: int,
    raw_msg_dim: int,
    memory_dim: int,
    time_dim: int,
    num_classes: int,
    tgat_heads: int,
    node_feature_dim: int | None,
) -> BaseTemporalModel:
    model_name = model_name.lower()

    if model_name == "tgn":
        return TGNClassifier(
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            num_classes=num_classes,
            node_feature_dim=node_feature_dim,
        )
    if model_name == "tgat":
        return TGATClassifier(
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            num_classes=num_classes,
            heads=tgat_heads,
            node_feature_dim=node_feature_dim,
        )
    if model_name == "dyrep":
        return DyRepClassifier(
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            num_classes=num_classes,
            node_feature_dim=node_feature_dim,
        )
    if model_name == "evolvegcn":
        return EvolveGCNOClassifier(
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            num_classes=num_classes,
            node_feature_dim=node_feature_dim,
        )
    if model_name == "sage":
        return GraphSAGEClassifier(
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            num_classes=num_classes,
            node_feature_dim=node_feature_dim,
        )

    raise ValueError(f"Unsupported model: {model_name}")
