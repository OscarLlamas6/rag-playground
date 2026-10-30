from app.domain import Chunk
from app.embeddings import embed, embed_query
from app.llm import NOT_FOUND, generate
from app.rag import cited_numbers
from app.store import MemoryStore


def make_chunks():
    texts = {"a": "tarifa del plan familiar mensual", "b": "horario de soporte telefónico", "c": "cancelar el contrato"}
    vectors = embed(list(texts.values()))
    return [Chunk(id=i, text=t, metadata={}, vector=v) for (i, t), v in zip(texts.items(), vectors)]


def test_memory_store_returns_most_similar_first():
    store = MemoryStore()
    chunks = make_chunks()
    store.reset(len(chunks[0].vector))
    store.upsert(chunks)
    hits = store.search(embed_query("cuál es la tarifa del plan familiar"), k=2)
    assert [h.id for h in hits][0] == "a"
    assert hits[0].score >= hits[1].score


def test_upsert_replaces_same_id():
    store = MemoryStore()
    chunks = make_chunks()
    store.reset(len(chunks[0].vector))
    store.upsert(chunks)
    store.upsert(chunks[:1])
    assert store.count() == 3


def test_fake_embeddings_are_unit_vectors():
    v = embed_query("hola mundo")
    assert abs(sum(x * x for x in v) - 1.0) < 1e-6


def test_cited_numbers():
    assert cited_numbers("Cuesta 39 [1] y dura 12 meses [2, 3].") == {1, 2, 3}
    assert cited_numbers("sin citas") == set()


def test_fake_llm_cites_or_abstains():
    prompt = ("<contexto>\n[1] (fuente: a.md · sección: Planes)\nEl plan Familiar cuesta 39 dólares al mes.\n\n</contexto>\n\n"
              "Pregunta del cliente: ¿Cuánto cuesta el plan Familiar?")
    assert "[1]" in generate(prompt).text
    other = prompt.replace("¿Cuánto cuesta el plan Familiar?", "¿Capital de Australia?")
    assert generate(other).text == NOT_FOUND
