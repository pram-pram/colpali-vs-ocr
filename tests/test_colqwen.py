import pytest
import torch
from fixtures import make_pages, synthetic_corpus
from PIL import Image

from cvo.data import Query
from cvo.retrievers.colqwen import ColQwen2Encoder, ColQwenRetriever, maxsim

DIM = 256


def _one_hot(i: int) -> torch.Tensor:
    v = torch.zeros(DIM, dtype=torch.float16)
    v[i % DIM] = 1.0
    return v


class StubEncoder:
    """Page embedding = [one_hot(red), 0.1 * ones]; query "pN" -> [one_hot(N)]."""

    model_id = "stub"

    def __init__(self, oom_above: int | None = None) -> None:
        self.count = 0
        self.oom_above = oom_above

    def encode_images(self, images: list[Image.Image]) -> list[torch.Tensor]:
        if self.oom_above is not None and len(images) > self.oom_above:
            raise torch.cuda.OutOfMemoryError("stub OOM")
        self.count += len(images)
        return [
            torch.stack([_one_hot(im.getpixel((0, 0))[0]), torch.full((DIM,), 0.1, dtype=torch.float16)])
            for im in images
        ]

    def encode_queries(self, texts: list[str]) -> list[torch.Tensor]:
        return [_one_hot(int(t[1:])).unsqueeze(0) for t in texts]


def test_maxsim_hand_computed():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    d = [torch.tensor([[1.0, 0.0], [0.5, 0.5]]), torch.tensor([[0.0, 2.0]])]
    assert maxsim(q, d).tolist() == [1.5, 2.0]


def test_resume_skips_completed_shards(tmp_path):  # Review Focus 4
    pages = make_pages(250); enc = StubEncoder()
    ColQwenRetriever(enc, tmp_path, shard_size=100).index(pages[:100])
    enc.count = 0
    stats = ColQwenRetriever(enc, tmp_path, shard_size=100).index(pages)
    assert enc.count == 150 and stats.num_pages == 250


def test_shard_with_different_pages_is_recomputed(tmp_path):  # Review Focus 4
    enc = StubEncoder()
    ColQwenRetriever(enc, tmp_path, shard_size=100).index(make_pages(50))   # --limit 50 run
    enc.count = 0
    ColQwenRetriever(enc, tmp_path, shard_size=100).index(make_pages(100))
    assert enc.count == 100


def test_oom_halves_batch(tmp_path):
    enc = StubEncoder(oom_above=1)
    r = ColQwenRetriever(enc, tmp_path, batch_size=4); r.index(make_pages(5))
    assert r.batch_size == 1


def test_oom_at_batch_size_one_reraises(tmp_path):
    r = ColQwenRetriever(StubEncoder(oom_above=0), tmp_path, batch_size=4)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        r.index(make_pages(2))


def test_stats_and_search(tmp_path):
    enc = StubEncoder()
    r = ColQwenRetriever(enc, tmp_path, shard_size=4)
    assert r.name == "colqwen" and r.model_id == "stub"
    stats = r.index(make_pages(10))
    assert stats.num_pages == 10 and stats.pages_failed == 0 and stats.seconds >= 0
    assert stats.bytes_stored == 10 * 2 * DIM * 2  # 2 rows x DIM fp16 per page
    assert sorted(p.name for p in tmp_path.glob("shard_*.pt")) == [
        "shard_00000.pt", "shard_00001.pt", "shard_00002.pt"
    ]
    queries = [Query(id="a", text="p3", relevant_page_ids=frozenset({"p3"})),
               Query(id="b", text="p7", relevant_page_ids=frozenset({"p7"}))]
    top = r.search(queries, k=3)
    assert [pid for pid, _ in top["a"]][0] == "p3" and top["b"][0][0] == "p7"
    assert len(top["a"]) == 3
    assert all(s1 >= s2 for (_, s1), (_, s2) in zip(top["a"], top["a"][1:]))


def test_search_after_full_reuse_from_disk(tmp_path):
    ColQwenRetriever(StubEncoder(), tmp_path, shard_size=4).index(make_pages(10))
    enc = StubEncoder()
    r = ColQwenRetriever(enc, tmp_path, shard_size=4)
    stats = r.index(make_pages(10))
    assert enc.count == 0 and stats.bytes_stored == 10 * 2 * DIM * 2
    top = r.search([Query(id="a", text="p5", relevant_page_ids=frozenset({"p5"}))], k=1)
    assert top["a"][0][0] == "p5"


@pytest.mark.gpu
def test_real_colqwen_smoke(tmp_path):
    pages, queries, _ = synthetic_corpus()
    r = ColQwenRetriever(ColQwen2Encoder(), tmp_path); r.index(pages)
    assert len(r.search(queries, k=3)) == len(queries)
