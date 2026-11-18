package ma.emsi.namous.tp4.tests;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;

import java.util.Map;
import java.util.List;
import java.util.Scanner;

public class TestRoutage {
    static void main(String[] args) {
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document documentRag = ClassPathDocumentLoader.loadDocument("rag.pdf", documentParser);
        Document documentAeronefs = ClassPathDocumentLoader.loadDocument("aeronefs.pdf", documentParser);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        EmbeddingStore<TextSegment> embeddingStoreRag = createEmbeddingStore(documentRag, embeddingModel);
        EmbeddingStore<TextSegment> embeddingStoreAeronefs = createEmbeddingStore(documentAeronefs, embeddingModel);

        ContentRetriever retrieverRag = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStoreRag)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        ContentRetriever retrieverAeronefs = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStoreAeronefs)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.3)
                .logRequests(true)
                .logResponses(true)
                .build();


        Map<ContentRetriever, String> retrievers = Map.of(
                retrieverRag,
                "Support de cours sur l'intelligence artificielle, "
                        + "le fine-tuning, le RAG et les modèles de langage.",

                retrieverAeronefs,
                "Document sur les aéronefs, les avions "
                        + "et les caractéristiques de l'aviation."
        );

        QueryRouter queryRouter = new LanguageModelQueryRouter(model, retrievers);
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        // Code que j'ai pris de TP2
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question : ");
                String question = scanner.nextLine();
                if (question.isBlank()) {
                    continue;
                }
                System.out.println("==================================================");
                if ("fin".equalsIgnoreCase(question)) {
                    break;
                }
                String reponse = assistant.chat(question);
                System.out.println("Assistant : " + reponse);
                System.out.println("==================================================");
            }
        }

    }

    private static EmbeddingStore<TextSegment> createEmbeddingStore(
            Document document, EmbeddingModel embeddingModel) {
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 20);
        List<TextSegment> segments = splitter.split(document);
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        return embeddingStore;
    }
}
