import asyncio
import platform
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util

class PDFRAGSystem:
    def __init__(self):
        self.vector_db = None
        self.retriever = None
        self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")

    def _evaluate_accuracy(self, data: dict) -> dict:
        """Evaluate answer quality using semantic similarity and lightweight keyword overlap"""
        answer = data["answer"]
        context = " ".join([doc.page_content for doc in data["context"]])
        
        # Semantic similarity
        answer_embedding = self.sentence_transformer.encode(answer, convert_to_tensor=True)
        context_embedding = self.sentence_transformer.encode(context, convert_to_tensor=True)
        semantic_similarity = util.cos_sim(answer_embedding, context_embedding).item()
        
        # Lightweight keyword overlap
        answer_keywords = set(answer.lower().split())
        context_keywords = set(context.lower().split())
        keyword_overlap = len(answer_keywords.intersection(context_keywords)) / max(len(answer_keywords), 1.0)
        
        # Combined confidence score
        confidence = 0.7 * semantic_similarity + 0.3 * keyword_overlap
        
        # Determine error type
        if confidence > 0.85:
            error_type = "None"
        elif confidence > 0.65:
            error_type = "Partial (Possible Type 2)"
        else:
            error_type = "Type 1 (High Risk)"
            
        return {
            "answer": answer,
            "confidence": confidence,
            "error_type": error_type,
            "context_snippets": [f"Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:200]}..." for doc in data["context"]]
        }

    def _get_prompt_template(self, language):
        """Return a prompt optimized for indirect and case-study questions with reasoning"""
        lang_instructions = {
            "English": "Answer ONLY in English.",
            "Hindi": "उत्तर केवल हिंदी में दें।",
            "Spanish": "Responde SOLO en español.",
            "French": "Répondez UNIQUEMENT en français.",
            "German": "Antworte NUR auf Deutsch."
        }

        templates = {
            "English": """You are an expert analyst answering questions based ONLY on the provided context from a PDF document. For indirect, reasoning-based, or case-study questions, use concise reasoning to synthesize information. If the context lacks explicit information, provide a logical answer based on general principles, clearly stating any assumptions and noting that the response is not directly from the context. If unsure, state: "I couldn't find a definitive answer in the provided context."

**Examples**:
1. **Question**: How can the document's strategies apply to a new market?
   **Answer**: Identify key strategies (e.g., pricing), consider market factors, and suggest adaptations (e.g., adjust pricing for local demand). [Concise response based on context]

2. **Question**: Which organizational model supports specialization and communication?
   **Answer**: If the context mentions departments, a matrix model may work, balancing functional specialization with cross-departmental teams. Without explicit context, a matrix model is logical for combining expertise and collaboration. [Reasoned response with assumption]

3. **Question**: Analyze the case study and suggest improvements.
   **Answer**: Summarize the case (problem, solution), note weaknesses, and propose specific improvements (e.g., enhance resource allocation). [Concise response based on context]

**Context**:
{context}

**Question**: {question}

**Guidelines**:
- Be precise, factual, and concise.
- For indirect or case-study questions, reason step-by-step and state assumptions if context is insufficient.
- Cite page numbers when possible.
- {lang_instruction}

**Answer**:""",
            
            "Hindi": """आप एक विशेषज्ञ विश्लेषक हैं जो पीडीएफ दस्तावेज़ के संदर्भ के आधार पर सवालों का जवाब देते हैं। अप्रत्यक्ष, तर्क-आधारित, या केस स्टडी सवालों के लिए, संक्षिप्त तर्क का उपयोग करके जानकारी संश्लेषित करें। यदि संदर्भ में स्पष्ट जानकारी नहीं है, तो सामान्य सिद्धांतों के आधार पर तार्किक जवाब दें, स्पष्ट रूप से कोई भी धारणाएँ बताते हुए और यह उल्लेख करें कि जवाब सीधे संदर्भ से नहीं है। यदि अनिश्चित हैं, तो कहें: "मुझे संदर्भ में स्पष्ट जवाब नहीं मिला।"

**उदाहरण**:
1. **प्रश्न**: दस्तावेज़ की रणनीतियों को नए बाजार में कैसे लागू किया जा सकता है?
   **उत्तर**: प्रमुख रणनीतियों (जैसे मूल्य निर्धारण) की पहचान करें, बाजार कारकों पर विचार करें, और अनुकूलन सुझाएं (जैसे स्थानीय मांग के लिए मूल्य समायोजन)। [संदर्भ के आधार पर संक्षिप्त जवाब]

2. **प्रश्न**: कौन सा संगठनात्मक मॉडल विशेषज्ञता और संचार का समर्थन करता है?
   **उत्तर**: यदि संदर्भ में विभागों का उल्लेख है, तो मैट्रिक्स मॉडल कार्य कर सकता है, जो कार्यात्मक विशेषज्ञता को क्रॉस-डिपार्टमेंटल टीमों के साथ संतुलित करता है। बिना स्पष्ट संदर्भ के, मैट्रिक्स मॉडल विशेषज्ञता और सहयोग के लिए तार्किक है। [तर्कपूर्ण जवाब धारणा के साथ]

3. **प्रश्न**: केस स्टडी का विश्लेषण करें और सुधार सुझाएं।
   **उत्तर**: केस का सारांश दें (समस्या, समाधान), कमजोरियों को नोट करें, और विशिष्ट सुधार सुझाएं (जैसे संसाधन आवंटन में सुधार)। [संदर्भ के आधार पर संक्षिप्त जवाब]

**संदर्भ**:
{context}

**प्रश्न**: {question}

**निर्देश**:
- सटीक, तथ्यात्मक और संक्षिप्त रहें।
- अप्रत्यक्ष या केस स्टडी सवालों के लिए, चरणबद्ध तर्क करें और यदि संदर्भ अपर्याप्त हो तो धारणाएँ बताएं।
- संभव हो तो पृष्ठ संख्या का उल्लेख करें।
- {lang_instruction}

**उत्तर**:""",
            
            "Spanish": """Eres un analista experto que responde preguntas basándote SOLO en el contexto de un documento PDF. Para preguntas indirectas, basadas en razonamiento o de estudios de caso, usa un razonamiento conciso para sintetizar información. Si el contexto no tiene información explícita, proporciona una respuesta lógica basada en principios generales, indicando claramente cualquier suposición y señalando que la respuesta no proviene directamente del contexto. Si no estás seguro, di: "No encontré una respuesta definitiva en el contexto proporcionado."

**Ejemplos**:
1. **Pregunta**: ¿Cómo se pueden aplicar las estrategias del documento a un nuevo mercado?
   **Respuesta**: Identifica estrategias clave (por ejemplo, precios), considera factores del mercado y sugiere adaptaciones (por ejemplo, ajustar precios según la demanda local). [Respuesta concisa basada en el contexto]

2. **Pregunta**: ¿Qué modelo organizacional apoya la especialización y la comunicación?
   **Respuesta**: Si el contexto menciona departamentos, un modelo matricial puede funcionar, equilibrando la especialización funcional con equipos interdepartamentales. Sin contexto explícito, un modelo matricial es lógico para combinar experiencia y colaboración. [Respuesta razonada con suposición]

3. **Pregunta**: Analiza el estudio de caso y sugiere mejoras.
   **Respuesta**: Resume el caso (problema, solución), señala debilidades y propone mejoras específicas (por ejemplo, optimizar asignación de recursos). [Respuesta concisa basada en el contexto]

**Contexto**:
{context}

**Pregunta**: {question}

**Directrices**:
- Sé preciso, factual y conciso.
- Para preguntas indirectas o de estudios de caso, razona paso a paso y menciona suposiciones si el contexto es insuficiente.
- Cita números de página cuando sea posible.
- {lang_instruction}

**Respuesta**:""",
            
            "French": """Vous êtes un analyste expert répondant aux questions en vous basant UNIQUEMENT sur le contexte d’un document PDF. Pour les questions indirectes, basées sur le raisonnement ou les études de cas, utilisez un raisonnement concis pour synthétiser l’information. Si le contexte manque d’informations explicites, fournissez une réponse logique basée sur des principes généraux, en indiquant clairement toute supposition et en précisant que la réponse ne provient pas directement du contexte. Si vous n’êtes pas sûr, dites : « Je n’ai pas trouvé de réponse définitive dans le contexte fourni. »

**Exemples** :
1. **Question** : Comment les stratégies du document peuvent-elles être appliquées à un nouveau marché ?
   **Réponse** : Identifiez les stratégies clés (par exemple, tarification), tenez compte des facteurs du marché et suggérez des adaptations (par exemple, ajuster les prix à la demande locale). [Réponse concise basée sur le contexte]

2. **Question** : Quel modèle organisationnel favorise la spécialisation et la communication ?
   **Réponse** : Si le contexte mentionne des départements, un modèle matriciel peut convenir, équilibrant la spécialisation fonctionnelle avec des équipes inter-départementales. Sans contexte explicite, un modèle matriciel est logique pour combiner expertise et collaboration. [Réponse raisonnée avec supposition]

3. **Question** : Analysez l’étude de cas et suggérez des améliorations.
   **Réponse** : Résumez le cas (problème, solution), notez les faiblesses et proposez des améliorations spécifiques (par exemple, optimiser l’allocation des ressources). [Réponse concise basée sur le contexte]

**Contexte** :
{context}

**Question** : {question}

**Directives** :
- Soyez précis, factuel et concis.
- Pour les questions indirectes ou les études de cas, raisonnez étape par étape et mentionnez les suppositions si le contexte est insuffisant.
- Citez les numéros de page lorsque possible.
- {lang_instruction}

**Réponse** :""",
            
            "German": """Du bist ein Expertenanalyst, der Fragen AUSSCHLIEßLICH auf Basis des Kontexts eines PDF-Dokuments beantwortet. Für indirekte, auf Reasoning basierende oder Fallstudienfragen verwende prägnantes Denken, um Informationen zu synthetisieren. Wenn der Kontext keine expliziten Informationen enthält, gib eine logische Antwort basierend auf allgemeinen Prinzipien, stelle klar alle Annahmen und weise darauf hin, dass die Antwort nicht direkt aus dem Kontext stammt. Wenn du unsicher bist, sage: „Ich konnte im Kontext keine definitive Antwort finden.“

**Beispiele**:
1. **Frage**: Wie können die Strategien des Dokuments auf einen neuen Markt angewendet werden?
   **Antwort**: Identifiziere Schlüsselstrategien (z. B. Preisgestaltung), berücksichtige Marktfaktoren und schlage Anpassungen vor (z. B. Preis an lokale Nachfrage anpassen). [Prägnante Antwort basierend auf Kontext]

2. **Frage**: Welches Organisationsmodell unterstützt Spezialisierung und Kommunikation?
   **Antwort**: Wenn der Kontext Abteilungen erwähnt, kann ein Matrixmodell funktionieren, das funktionale Spezialisierung mit interdisziplinären Teams ausgleicht. Ohne expliziten Kontext ist ein Matrixmodell logisch für die Kombination von Expertise und Zusammenarbeit. [Begründete Antwort mit Annahme]

3. **Frage**: Analysiere die Fallstudie und schlage Verbesserungen vor.
   **Antwort**: Fasse den Fall zusammen (Problem, Lösung), notiere Schwächen und schlage spezifische Verbesserungen vor (z. B. Ressourcenzuweisung optimieren). [Prägnante Antwort basierend auf Kontext]

**Kontext**:
{context}

**Frage**: {question}

**Richtlinien**:
- Sei präzise, faktenbasiert und prägnant.
- Für indirekte oder Fallstudienfragen denke Schritt für Schritt und nenne Annahmen, wenn der Kontext unzureichend ist.
- Nenne Seitenzahlen, wenn möglich.
- {lang_instruction}

**Antwort**:"""
        }

        template = templates.get(language, templates["English"])
        return template.replace(
            "{lang_instruction}",
            lang_instructions.get(language, "Answer ONLY in English.")
        )

    def load_pdf(self, file_path):
        """Load and process PDF file into chunks and vector DB"""
        try:
            loader = PyPDFLoader(file_path)
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=300,
                add_start_index=True
            )
            chunks = text_splitter.split_documents(data)

            fast_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            self.vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=fast_embeddings,
                collection_name="pdf-rag"
            )

            self.retriever = self.vector_db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 15, "score_threshold": 0.4}  # Increased k, lowered threshold
            )

            return True, f"PDF processed successfully! {len(chunks)} chunks created."
        except Exception as e:
            return False, f"Error: {str(e)}"

    def ask_question(self, question, language="English"):
        """Ask a question with optimized reasoning for indirect and case-study questions"""
        if not self.retriever:
            return {"error": "Please upload and process a PDF first"}
                                                                                                    
        try:
            prompt_template = self._get_prompt_template(language)
            prompt = ChatPromptTemplate.from_template(prompt_template)

            llm = ChatOllama(model="llama3")

            base_chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            evaluation_chain = (
                {"answer": base_chain, "context": self.retriever}
                | RunnableLambda(self._evaluate_accuracy)
            )

            if platform.system() == "Emscripten":
                async def run_chain():
                    return await asyncio.get_event_loop().run_in_executor(None, evaluation_chain.invoke, question)
                result = asyncio.run(run_chain())
            else:
                result = evaluation_chain.invoke(question)

            return {
                "answer": result["answer"],
                "confidence": f"{result['confidence']*100:.1f}%",
                "error_risk": result["error_type"],
                "context_snippets": result["context_snippets"]
            }

        except Exception as e:
            return {"error": f"Error generating answer: {str(e)}"}

if platform.system() == "Emscripten":
    async def main():
        rag_system = PDFRAGSystem()
        await asyncio.sleep(0)
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        rag_system = PDFRAGSystem()