import importlib
from typing import Dict, Optional, Union

from spmem_memory.configs.embeddings.base import BaseEmbedderConfig
from spmem_memory.configs.llms.anthropic import AnthropicConfig
from spmem_memory.configs.llms.azure import AzureOpenAIConfig
from spmem_memory.configs.llms.base import BaseLlmConfig
from spmem_memory.configs.llms.deepseek import DeepSeekConfig
from spmem_memory.configs.llms.lmstudio import LMStudioConfig
from spmem_memory.configs.llms.ollama import OllamaConfig
from spmem_memory.configs.llms.openai import OpenAIConfig
from spmem_memory.configs.llms.vllm import VllmConfig
from spmem_memory.configs.rerankers.base import BaseRerankerConfig
from spmem_memory.configs.rerankers.cohere import CohereRerankerConfig
from spmem_memory.configs.rerankers.sentence_transformer import SentenceTransformerRerankerConfig
from spmem_memory.configs.rerankers.zero_entropy import ZeroEntropyRerankerConfig
from spmem_memory.configs.rerankers.llm import LLMRerankerConfig
from spmem_memory.configs.rerankers.huggingface import HuggingFaceRerankerConfig
from spmem_memory.embeddings.mock import MockEmbeddings


def load_class(class_type):
    module_path, class_name = class_type.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class LlmFactory:
    """
    Factory for creating LLM instances with appropriate configurations.
    Supports both old-style BaseLlmConfig and new provider-specific configs.
    """

    # Provider mappings with their config classes
    provider_to_class = {
        "ollama": ("spmem_memory.llms.ollama.OllamaLLM", OllamaConfig),
        "openai": ("spmem_memory.llms.openai.OpenAILLM", OpenAIConfig),
        "groq": ("spmem_memory.llms.groq.GroqLLM", BaseLlmConfig),
        "together": ("spmem_memory.llms.together.TogetherLLM", BaseLlmConfig),
        "aws_bedrock": ("spmem_memory.llms.aws_bedrock.AWSBedrockLLM", BaseLlmConfig),
        "litellm": ("spmem_memory.llms.litellm.LiteLLM", BaseLlmConfig),
        "azure_openai": ("spmem_memory.llms.azure_openai.AzureOpenAILLM", AzureOpenAIConfig),
        "openai_structured": ("spmem_memory.llms.openai_structured.OpenAIStructuredLLM", OpenAIConfig),
        "anthropic": ("spmem_memory.llms.anthropic.AnthropicLLM", AnthropicConfig),
        "azure_openai_structured": ("spmem_memory.llms.azure_openai_structured.AzureOpenAIStructuredLLM", AzureOpenAIConfig),
        "gemini": ("spmem_memory.llms.gemini.GeminiLLM", BaseLlmConfig),
        "deepseek": ("spmem_memory.llms.deepseek.DeepSeekLLM", DeepSeekConfig),
        "xai": ("spmem_memory.llms.xai.XAILLM", BaseLlmConfig),
        "sarvam": ("spmem_memory.llms.sarvam.SarvamLLM", BaseLlmConfig),
        "lmstudio": ("spmem_memory.llms.lmstudio.LMStudioLLM", LMStudioConfig),
        "vllm": ("spmem_memory.llms.vllm.VllmLLM", VllmConfig),
        "langchain": ("spmem_memory.llms.langchain.LangchainLLM", BaseLlmConfig),
    }

    @classmethod
    def create(cls, provider_name: str, config: Optional[Union[BaseLlmConfig, Dict]] = None, **kwargs):
        """
        Create an LLM instance with the appropriate configuration.

        Args:
            provider_name (str): The provider name (e.g., 'openai', 'anthropic')
            config: Configuration object or dict. If None, will create default config
            **kwargs: Additional configuration parameters

        Returns:
            Configured LLM instance

        Raises:
            ValueError: If provider is not supported
        """
        if provider_name not in cls.provider_to_class:
            raise ValueError(f"Unsupported Llm provider: {provider_name}")

        class_type, config_class = cls.provider_to_class[provider_name]
        llm_class = load_class(class_type)

        # Handle configuration
        if config is None:
            # Create default config with kwargs
            config = config_class(**kwargs)
        elif isinstance(config, dict):
            # Merge dict config with kwargs
            config.update(kwargs)
            config = config_class(**config)
        elif isinstance(config, BaseLlmConfig):
            # Convert base config to provider-specific config if needed
            if config_class != BaseLlmConfig:
                # Convert to provider-specific config
                config_dict = {
                    "model": config.model,
                    "temperature": config.temperature,
                    "api_key": config.api_key,
                    "max_tokens": config.max_tokens,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "enable_vision": config.enable_vision,
                    "vision_details": config.vision_details,
                    "http_client_proxies": config.http_client,
                }
                config_dict.update(kwargs)
                config = config_class(**config_dict)
            else:
                # Use base config as-is
                pass
        else:
            # Assume it's already the correct config type
            pass

        return llm_class(config)

    @classmethod
    def register_provider(cls, name: str, class_path: str, config_class=None):
        """
        Register a new provider.

        Args:
            name (str): Provider name
            class_path (str): Full path to LLM class
            config_class: Configuration class for the provider (defaults to BaseLlmConfig)
        """
        if config_class is None:
            config_class = BaseLlmConfig
        cls.provider_to_class[name] = (class_path, config_class)

    @classmethod
    def get_supported_providers(cls) -> list:
        """
        Get list of supported providers.

        Returns:
            list: List of supported provider names
        """
        return list(cls.provider_to_class.keys())


class EmbedderFactory:
    provider_to_class = {
        "openai": "spmem_memory.embeddings.openai.OpenAIEmbedding",
        "ollama": "spmem_memory.embeddings.ollama.OllamaEmbedding",
        "huggingface": "spmem_memory.embeddings.huggingface.HuggingFaceEmbedding",
        "azure_openai": "spmem_memory.embeddings.azure_openai.AzureOpenAIEmbedding",
        "gemini": "spmem_memory.embeddings.gemini.GoogleGenAIEmbedding",
        "vertexai": "spmem_memory.embeddings.vertexai.VertexAIEmbedding",
        "together": "spmem_memory.embeddings.together.TogetherEmbedding",
        "lmstudio": "spmem_memory.embeddings.lmstudio.LMStudioEmbedding",
        "langchain": "spmem_memory.embeddings.langchain.LangchainEmbedding",
        "aws_bedrock": "spmem_memory.embeddings.aws_bedrock.AWSBedrockEmbedding",
        "fastembed": "spmem_memory.embeddings.fastembed.FastEmbedEmbedding",
    }

    @classmethod
    def create(cls, provider_name, config, vector_config: Optional[dict]):
        if provider_name == "upstash_vector" and vector_config and vector_config.enable_embeddings:
            return MockEmbeddings()
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            embedder_instance = load_class(class_type)
            base_config = BaseEmbedderConfig(**config)
            return embedder_instance(base_config)
        else:
            raise ValueError(f"Unsupported Embedder provider: {provider_name}")


class VectorStoreFactory:
    provider_to_class = {
        "qdrant": "spmem_memory.vector_stores.qdrant.Qdrant",
        "chroma": "spmem_memory.vector_stores.chroma.ChromaDB",
        "pgvector": "spmem_memory.vector_stores.pgvector.PGVector",
        "milvus": "spmem_memory.vector_stores.milvus.MilvusDB",
        "upstash_vector": "spmem_memory.vector_stores.upstash_vector.UpstashVector",
        "azure_ai_search": "spmem_memory.vector_stores.azure_ai_search.AzureAISearch",
        "azure_mysql": "spmem_memory.vector_stores.azure_mysql.AzureMySQL",
        "pinecone": "spmem_memory.vector_stores.pinecone.PineconeDB",
        "mongodb": "spmem_memory.vector_stores.mongodb.MongoDB",
        "redis": "spmem_memory.vector_stores.redis.RedisDB",
        "valkey": "spmem_memory.vector_stores.valkey.ValkeyDB",
        "databricks": "spmem_memory.vector_stores.databricks.Databricks",
        "elasticsearch": "spmem_memory.vector_stores.elasticsearch.ElasticsearchDB",
        "vertex_ai_vector_search": "spmem_memory.vector_stores.vertex_ai_vector_search.GoogleMatchingEngine",
        "opensearch": "spmem_memory.vector_stores.opensearch.OpenSearchDB",
        "supabase": "spmem_memory.vector_stores.supabase.Supabase",
        "weaviate": "spmem_memory.vector_stores.weaviate.Weaviate",
        "faiss": "spmem_memory.vector_stores.faiss.FAISS",
        "langchain": "spmem_memory.vector_stores.langchain.Langchain",
        "s3_vectors": "spmem_memory.vector_stores.s3_vectors.S3Vectors",
        "baidu": "spmem_memory.vector_stores.baidu.BaiduDB",
        "cassandra": "spmem_memory.vector_stores.cassandra.CassandraDB",
        "neptune": "spmem_memory.vector_stores.neptune_analytics.NeptuneAnalyticsVector",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name)
        if class_type:
            if not isinstance(config, dict):
                config = config.model_dump()
            vector_store_instance = load_class(class_type)
            return vector_store_instance(**config)
        else:
            raise ValueError(f"Unsupported VectorStore provider: {provider_name}")

    @classmethod
    def reset(cls, instance):
        instance.reset()
        return instance


class GraphStoreFactory:
    """
    Factory for creating MemoryGraph instances for different graph store providers.
    Usage: GraphStoreFactory.create(provider_name, config)
    """

    provider_to_class = {
        "memgraph": "spmem_memory.memory.memgraph_memory.MemoryGraph",
        "neptune": "spmem_memory.graphs.neptune.neptunegraph.MemoryGraph",
        "neptunedb": "spmem_memory.graphs.neptune.neptunedb.MemoryGraph",
        "kuzu": "spmem_memory.memory.kuzu_memory.MemoryGraph",
        "default": "spmem_memory.memory.graph_memory.MemoryGraph",
    }

    @classmethod
    def create(cls, provider_name, config):
        class_type = cls.provider_to_class.get(provider_name, cls.provider_to_class["default"])
        try:
            GraphClass = load_class(class_type)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import MemoryGraph for provider '{provider_name}': {e}")
        return GraphClass(config)


class RerankerFactory:
    """
    Factory for creating reranker instances with appropriate configurations.
    Supports provider-specific configs following the same pattern as other factories.
    """

    # Provider mappings with their config classes
    provider_to_class = {
        "cohere": ("spmem_memory.reranker.cohere_reranker.CohereReranker", CohereRerankerConfig),
        "sentence_transformer": ("spmem_memory.reranker.sentence_transformer_reranker.SentenceTransformerReranker", SentenceTransformerRerankerConfig),
        "zero_entropy": ("spmem_memory.reranker.zero_entropy_reranker.ZeroEntropyReranker", ZeroEntropyRerankerConfig),
        "llm_reranker": ("spmem_memory.reranker.llm_reranker.LLMReranker", LLMRerankerConfig),
        "huggingface": ("spmem_memory.reranker.huggingface_reranker.HuggingFaceReranker", HuggingFaceRerankerConfig),
    }

    @classmethod
    def create(cls, provider_name: str, config: Optional[Union[BaseRerankerConfig, Dict]] = None, **kwargs):
        """
        Create a reranker instance based on the provider and configuration.

        Args:
            provider_name: The reranker provider (e.g., 'cohere', 'sentence_transformer')
            config: Configuration object or dictionary
            **kwargs: Additional configuration parameters

        Returns:
            Reranker instance configured for the specified provider

        Raises:
            ImportError: If the provider class cannot be imported
            ValueError: If the provider is not supported
        """
        if provider_name not in cls.provider_to_class:
            raise ValueError(f"Unsupported reranker provider: {provider_name}")

        class_path, config_class = cls.provider_to_class[provider_name]

        # Handle configuration
        if config is None:
            config = config_class(**kwargs)
        elif isinstance(config, dict):
            config = config_class(**config, **kwargs)
        elif not isinstance(config, BaseRerankerConfig):
            raise ValueError(f"Config must be a {config_class.__name__} instance or dict")

        # Import and create the reranker class
        try:
            reranker_class = load_class(class_path)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import reranker for provider '{provider_name}': {e}")

        return reranker_class(config)
