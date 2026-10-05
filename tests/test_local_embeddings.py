"""Provider routing and local embedding response validation."""

import httpx
import pytest

from backend.app.core.config import Settings
from backend.app.services.local_embeddings import OllamaEmbeddings


def test_local_index_is_separate_and_openai_setting_is_preserved():
    baseline = Settings(embedding_provider='openai')
    local = Settings(embedding_provider='ollama')
    assert baseline.embedding_index_directory == baseline.chroma_persist_directory
    assert local.embedding_index_directory != baseline.embedding_index_directory
    assert local.embedding_model == 'bge-m3'
    assert local.openai_embedding_model == baseline.openai_embedding_model
    assert local.embedding_index_directory != Settings(embedding_provider='ollama', ollama_embedding_model='other').embedding_index_directory


@pytest.mark.parametrize('vectors,valid', [([[1., 2.], [3., 4.]], True),
    ([[1.]], False), ([[1.], [2., 3.]], False), ([[True], [2.]], False),
    ([[float('inf')], [2.]], False)])
def test_local_embedding_batch(monkeypatch, vectors, valid):
    def post(client, url, *, json):
        assert url == 'http://localhost:11434/api/embed'
        assert json['model'] == 'bge-m3'
        assert json['input'] == ['first', 'second']
        assert json['truncate'] is False
        return type('Response', (), {'raise_for_status': lambda self: None,
            'json': lambda self: {'embeddings': vectors}})()
    monkeypatch.setattr(httpx.Client, 'post', post)
    model = OllamaEmbeddings('bge-m3', 'http://localhost:11434/v1')
    if valid:
        assert model.embed_documents(['first', 'second']) == vectors
    else:
        with pytest.raises(ValueError):
            model.embed_documents(['first', 'second'])


def test_query_uses_same_model_and_http_errors_do_not_fallback(monkeypatch):
    model = OllamaEmbeddings('bge-m3', 'http://localhost:11434/v1')
    monkeypatch.setattr(model, 'embed_documents', lambda texts: [[1., 2.]])
    assert model.embed_query('question') == [1., 2.]
    def fail(*args, **kwargs):
        raise httpx.ConnectError('offline')
    monkeypatch.setattr(httpx.Client, 'post', fail)
    with pytest.raises(httpx.ConnectError):
        OllamaEmbeddings('bge-m3', 'http://localhost:11434/v1').embed_query('question')
