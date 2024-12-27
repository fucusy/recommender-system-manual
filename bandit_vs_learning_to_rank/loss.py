import torch

def batch_pairs(x: torch.Tensor) -> torch.Tensor:
    """Returns a pair matrix

    This matrix contains all pairs (i, j) as follows:
        p[_, i, j, 0] = x[_, i]
        p[_, i, j, 1] = x[_, j]

    Args:
        x: The input batch of dimension (batch_size, list_size) or
            (batch_size, list_size, 1).

    Returns:
        Two tensors of size (batch_size, list_size ^ 2, 2) containing
        all pairs.
    """

    if x.dim() == 2:
        x = x.reshape((x.shape[0], x.shape[1], 1))

    # Construct broadcasted x_{:,i,0...list_size}
    x_ij = torch.repeat_interleave(x, x.shape[1], dim=2)

    # Construct broadcasted x_{:,0...list_size,i}
    x_ji = torch.repeat_interleave(x.permute(0, 2, 1), x.shape[1], dim=1)

    return torch.stack([x_ij, x_ji], dim=3)


class _PairwiseAdditiveLoss(torch.nn.Module):
    """Pairwise additive ranking losses.

    Implementation of linearly decomposible additive pairwise ranking losses.
    This includes RankSVM hinge loss and variations.
    """
    def __init__(self):
        r""""""
        super().__init__()

    def _loss_per_doc_pair(self, score_pairs: torch.FloatTensor,
                           rel_pairs: torch.LongTensor) -> torch.FloatTensor:
        """Computes a loss on given score pairs and relevance pairs.

        Args:
            score_pairs: A tensor of shape (batch_size, list_size,
                list_size, 2), where each entry (:, i, j, :) indicates a pair
                of scores for doc i and j.
            rel_pairs: A tensor of shape (batch_size, list_size, list_size, 2),
                where each entry (:, i, j, :) indicates the relevance
                for doc i and j.

        Returns:
            A tensor of shape (batch_size, list_size, list_size) with a loss
            per document pair.
        """
        raise NotImplementedError

    def _loss_reduction(self,
                        loss_pairs: torch.FloatTensor) -> torch.FloatTensor:
        """Reduces the paired loss to a per sample loss.

        Args:
            loss_pairs: A tensor of shape (batch_size, list_size, list_size)
                where each entry indicates the loss of doc pair i and j.

        Returns:
            A tensor of shape (batch_size) indicating the loss per training
            sample.
        """
        return loss_pairs.view(loss_pairs.shape[0], -1).sum(1)

    def _loss_modifier(self, loss: torch.FloatTensor) -> torch.FloatTensor:
        """A modifier to apply to the loss."""
        return loss

    def forward(self, scores: torch.FloatTensor, relevance: torch.LongTensor,
                n: torch.LongTensor) -> torch.FloatTensor:
        """Computes the loss for given batch of samples.

        Args:
            scores: A batch of per-query-document scores.
            relevance: A batch of per-query-document relevance labels.
            n: A batch of per-query number of documents (for padding purposes).
        """
        # Reshape relevance if necessary.
        if relevance.ndimension() == 2:
            relevance = relevance.reshape(
                (relevance.shape[0], relevance.shape[1], 1))
        if scores.ndimension() == 2:
            scores = scores.reshape((scores.shape[0], scores.shape[1], 1))

        # Compute pairwise differences for scores and relevances.
        score_pairs = batch_pairs(scores)
        rel_pairs = batch_pairs(relevance)

        # Compute loss per doc pair.
        loss_pairs = self._loss_per_doc_pair(score_pairs, rel_pairs)

        # Mask out padded documents per query in the batch
        n_grid = n[:, None, None].repeat(1, score_pairs.shape[1],
                                         score_pairs.shape[2])
        arange = torch.arange(score_pairs.shape[1],
                               device=score_pairs.device)
        range_grid = torch.max(*torch.meshgrid([arange, arange]))
        range_grid = range_grid[None, :, :].repeat(n.shape[0], 1, 1)
        loss_pairs[n_grid <= range_grid] = 0.0

        # Reduce final list loss from per doc pair loss to a per query loss.
        loss = self._loss_reduction(loss_pairs)

        # Apply a loss modifier.
        loss = self._loss_modifier(loss)

        # Return loss
        return loss


class PairwiseHingeLoss(_PairwiseAdditiveLoss):
    r"""Pairwise hinge loss formulation of SVMRank:

    .. math::
        l(\mathbf{s}, \mathbf{y}) = \sum_{y_i > y _j} max\left(
        0, 1 - (s_i - s_j)
        \right)

    Shape:
        - input scores: :math:`(N, \texttt{list_size})`
        - input relevance: :math:`(N, \texttt{list_size})`
        - input n: :math:`(N)`
        - output: :math:`(N)`
    """
    def _loss_per_doc_pair(self, score_pairs, rel_pairs):
        score_pair_diffs = score_pairs[:, :, :, 0] - score_pairs[:, :, :, 1]
        rel_pair_diffs = rel_pairs[:, :, :, 0] - rel_pairs[:, :, :, 1]
        loss = 1.0 - score_pair_diffs
        loss[rel_pair_diffs <= 0.0] = 0.0
        loss[loss < 0.0] = 0.0
        return loss



class TweedieLoss(torch.nn.Module):
    """
    Tweedie loss.

    Tweedie regression with log-link. It might be useful, e.g., for modeling total
    loss in insurance, or for any target that might be tweedie-distributed.
    """

    def __init__(self, p: float = 1.5):
        """
        Args:
            p (float, optional): tweedie variance power which is greater equal
                1.0 and smaller 2.0. Close to ``2`` shifts to
                Gamma distribution and close to ``1`` shifts to Poisson distribution.
                Defaults to 1.5.
            reduction (str, optional): How to reduce the loss. Defaults to "mean".
        """
        super().__init__()
        assert 1 <= p < 2, "p must be in range [1, 2]"
        self.p = p

    def forward(self, y_pred, y_true):
        a = y_true * torch.exp(y_pred * (1 - self.p)) / (1 - self.p)
        b = torch.exp(y_pred * (2 - self.p)) / (2 - self.p)
        loss = -a + b
        return loss

if __name__ == "__main__":
    import torch

    def test_pairwise_hinge_loss():
        scores = torch.tensor([[0.2, 0.4, 0.6]], dtype=torch.float32)
        print(scores.size())
        relevance = torch.tensor([[1, 0, 2]], dtype=torch.float32)
        n = torch.tensor([3], dtype=torch.int32)

        loss_fn = PairwiseHingeLoss()
        loss = loss_fn(scores, relevance, n)
        print("Pairwise Hinge Loss:", loss)

    def test_tweedie_loss():
        y_true = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=torch.float32)
        y_pred = torch.tensor([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0], dtype=torch.float32)
        loss_fn = TweedieLoss(p=1.5)
        loss = loss_fn(y_pred, y_true)
        print("Tweedie Loss:", loss)

    test_pairwise_hinge_loss()
    test_tweedie_loss()