import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
# Pinecone setup
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "bonhoeffer-bot"
# Safety check
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not set in .env")
# Example: English Flyers as documents
docs = [
    """ Order Shipment Details


Order No: BP-61 (Incluida BP-62)

Products: DESBROZADORA, DESBROZADORA DE MOCHILA, MULTIFUNCIONAL & REPUESTOS
Shipment months/Production finish date: SOB - 24-MARCH-2025
Total Value: $101,094.94
Marketing Budget: $5,819.23
Payment received: $0.00
Remaining Balance: $95,275.71
Comments: ETA - 12-MAY-2025, PAGO ESPERADO


Order No: BP-62 (Incluida BP-53)

Products: FUMIGADORA & DESBROZADORA + POP
Shipment months/Production finish date: SOB - 30-MARCH-2025
Total Value: $75,404.20
Marketing Budget: $5,819.23
Payment received: $0.00
Remaining Balance: $69,584.97
Comments: ETA - 28-APRIL-2025, PAGO ESPERADO


Order No: BP-58

Products: MOTOSIERRA, ATOMIZADOR DE MOCHILA & CADENA
Shipment months/Production finish date: SOB - 03-APRIL-2025
Total Value: $150,576.00
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $150,576.00
Comments: ETA - 04-MAY-2025, PAGO ESPERADO


Order No: BP-59-II (Incluida BP-48 + BP-46 + BP-47 + BP-53 + BP-56)

Products: ATOMIZADOR DE MOCHILA, FUMIGADORA MANUAL, GENERADOR DIÉSEL SILENCIOSO, BOMBA DE AGUA DIESEL, MOTOR DIESEL, MOTOSIERRA & CADENA
Shipment months/Production finish date: SOB - 10-APRIL-2025
Total Value: $118,433.01
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $118,433.01
Comments: ETA - 12-MAY-2025


Order No: BP-60-I (Incluida BP-53)

Products: BOMBA DE AGUA, MOTOR, GENERADOR, MINI CULTIVADOR DE GASOLINA, MINI CULTIVADOR DE DIESEL, INVERSOR DE GASOLINA & ACCESORIOS PARA MINICULTIVADOR
Shipment months/Production finish date: SOB - 29-APRIL-2025
Total Value: $95,387.70
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $95,387.70
Comments: ETA - 22-JUNE-2025, PAGO ESPERADO


Order No: BP-60-II

Products: BOMBA DE AGUA, MOTOR, GENERADOR, MINI CULTIVADOR DE GASOLINA, MINI CULTIVADOR DE DIESEL, INVERSOR DE GASOLINA & ACCESORIOS PARA MINICULTIVADOR
Shipment months/Production finish date: SOB - 19-MAY-2025
Total Value: $90,165.52
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $90,165.52
Comments: ETA - 14-JULY-2025, PAGO ESPERADO


Order No: BP-76

Products: MOTOR ELÉCTRICO
Shipment months/Production finish date: 25-APRIL-2025
Total Value: $68,733.00
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $68,733.00
Comments: TOTAL CBM - 31.20, AGENT AWAITED


Order No: BP-48 (Incluida BP-47 + BP-61 + BP-53 + BP-56)

Products: MOTOSIERRA, BARRENA DE TIERRA, FUMIGADORA MANUAL, PULVERIZADOR DE MOCHILA, PODADORA PROFESIONAL & REPUESTOS
Shipment months/Production finish date: SOB - 04-MAY-2025
Total Value: $85,690.73
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $85,690.73
Comments: ETA - 06-JUNE-2025, PAGO ESPERADO


Order No: BP-64

Products: BOMBA DE AGUA, MOTOR, GENERADOR, MINI CULTIVADOR DE GASOLINA & ACCESORIOS PARA MINICULTIVADOR
Shipment months/Production finish date: JUNIO - 2025
Total Value: $77,784.70
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $77,784.70
Comments: EN PRODUCCIÓN, CBM - 65.95


Order No: BP-65

Products: BOMBA DE AGUA, MOTOR, GENERADOR, MINI CULTIVADOR DE GASOLINA, INVERSOR DE GASOLINA & ACCESORIOS PARA MINICULTIVADOR
Shipment months/Production finish date: JUNIO - 2025
Total Value: $126,083.70
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $126,083.70
Comments: EN PRODUCCIÓN, CBM - 65.47


Order No: BP-66

Products: BOMBA DE AGUA, MOTOR, GENERADOR, INVERSOR DE GASOLINA, MINI CULTIVADOR DE DIESEL & ACCESORIOS PARA MINICULTIVADOR
Shipment months/Production finish date: JUNIO - 2025
Total Value: $97,366.03
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $97,366.03
Comments: EN PRODUCCIÓN, CBM - 65.71


Order No: BP-69

Products: BOMBA DE AGUA, MOTOR, GENERADOR, INVERSOR DE GASOLINA, MINI CULTIVADOR DE DIESEL & ACCESORIOS PARA MINICULTIVADOR
Shipment months/Production finish date: JULIO - 2025
Total Value: $120,617.37
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $120,617.37
Comments: EN PRODUCCIÓN, CBM - 65.54


Order No: BP-71-A

Products: MOTOSIERRA, PULVERIZADOR DE MOCHILA, ATOMIZADOR DE MOCHILA, FUMIGADORA MANUAL, BOMBA SUMERGIBLE
Shipment months/Production finish date: JULIO - 2025
Total Value: $71,996.80
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $71,996.80
Comments: TOTAL CBM - 65.80


Order No: BP-71-B

Products: DESBROZADORA, PULVERIZADOR DE MOCHILA, FUMIGADORA MANUAL, BOMBA CENTRÍFUGA & PODADORA PROFESIONAL
Shipment months/Production finish date: JULIO - 2025
Total Value: $64,051.20
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $64,051.20
Comments: TOTAL CBM - 65.79


Order No: BP-0068

Products: MOTOR, GENERADOR, MINI CULTIVADOR
Shipment months/Production finish date: AGOSTO - 2025
Total Value: $90,924.15
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $90,924.15
Comments: EN PRODUCCIÓN, CBM - 65.15


Order No: BP-72

Products: DESBROZADORA, BARRENA DE TIERRA, MOTOSIERRA, PULVERIZADOR DE MOCHILA, FUMIGADORA MANUAL, LAVADORA A PRESIÓN, BOMBA SUMERGIBLE, CONDUCCIÓN DIRECTA COMPRESOR DE AIRE, MÁQUINAS DE SOLDAR, GENERADOR DIÉSEL SILENCIOSO
Shipment months/Production finish date: AGOSTO - 2025
Total Value: $103,409.48
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $103,409.48
Comments: TOTAL CBM - 66.18


Order No: BP-73

Products: DESBROZADORA, BARRENA DE TIERRA, PULVERIZADOR DE MOCHILA, FUMIGADORA MANUAL, BOMBA CENTRÍFUGA, PODADORA PROFESIONAL, CONDUCCIÓN DIRECTA COMPRESOR DE AIRE & MÁQUINAS DE SOLDAR
Shipment months/Production finish date: SEPTIEMBRE - 2025
Total Value: $73,245.60
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $73,245.60
Comments: TOTAL CBM - 65.54


Order No: BP-74

Products: BARRENA DE TIERRA, GENERADOR DIÉSEL SILENCIOSO, ATOMIZADOR DE MOCHILA, FUMIGADORA MANUAL, LAVADORA A PRESIÓN, BOMBA SUMERGIBLE, PODADORA PROFESIONAL
Shipment months/Production finish date: OCTUBRE - 2025
Total Value: $62,647.50
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $62,647.50
Comments: TOTAL CBM - 65.38


Order No: BP-75

Products: DESBROZADORA DE MOCHILA, MULTIFUNCIONAL, MOTOSIERRA, ATOMIZADOR DE MOCHILA, LAVADORA A PRESIÓN, BOMBA CENTRÍFUGA, BOMBA SUMERGIBLE, PODADORA PROFESIONAL, BARRENA DE TIERRA, MÁQUINAS DE SOLDAR
Shipment months/Production finish date: NOVIEMBRE - 2025
Total Value: $100,698.00
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $100,698.00
Comments: TOTAL CBM - 65.14


Order No: BP-0068-2

Products: INVERSOR DE GASOLINA
Shipment months/Production finish date: JULIO - 2025
Total Value: $26,820.00
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $26,820.00
Comments: TOTAL CBM - 9.28


Order No: BP-0078

Products: REPUESTOS
Shipment months/Production finish date: JUNIO-JULIO - 2025
Total Value: $45,530.64
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $45,530.64
Comments: TOTAL CBM - 15


Order No: BP-0080

Products: MOTORES DIESEL, MOLINO, GENERADOR, DESBROZADORA, PULVERIZADOR DE MOCHILA
Shipment months/Production finish date: ESPERADO
Total Value: $218,766.15
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $218,766.15
Comments: TOTAL CBM - 134


Order No: BP-0082

Products: GENERADORES, MOTORES
Shipment months/Production finish date: ESPERADO
Total Value: $121,353.55
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $121,353.55
Comments: TOTAL CBM - 81.2


Order No: BP-0070-I

Products: BOMBA CENTRÍFUGA, BARRENA, PULVERIZADOR, PULVERIZADOR MANUAL ELÉCTRICO, GENERADOR DIÉSEL, CORTACÉSPED, PULVERIZADOR MANUAL, DESBROZADORA
Shipment months/Production finish date: JULIO - 2025
Total Value: $61,752.70
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $61,752.70
Comments: TOTAL CBM - 65.79


Order No: BP-0070-II

Products: HIDROLAVADORA, BOMBA CENTRÍFUGA, BOMBA SUMERGIBLE, MÁQUINA DE SOLDAR, BARRENA, COMPRESOR DE AIRE, PULVERIZADOR MANUAL ELÉCTRICO, GENERADOR DIÉSEL, CORTADORA DE CÉSPED, PULVERIZADOR MANUAL
Shipment months/Production finish date: JULIO - 2025
Total Value: $70,266.50
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $70,266.50
Comments: TOTAL CBM - 65.79


Order No: BP-0067-I

Products: PULVERIZADOR DE NIEBLA, MOTOSIERRA, PULVERIZADOR DE MOCHILA, DESBROZADORA, MULTIHERRAMIENTA
Shipment months/Production finish date: JUNIO - 2025
Total Value: $65,184.00
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $65,184.00
Comments: AGENTE ESPERADO, TOTAL CBM - 65.79


Order No: BP-0067-II

Products: PULVERIZADOR MANUAL, PULVERIZADOR DE MOCHILA, DESBROZADORA, HERRAMIENTA MULTIUSOS
Shipment months/Production finish date: JULIO - 2025
Total Value: $67,613.90
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $67,613.90
Comments: TOTAL CBM - 65.79


Order No: BP-0063-I

Products: CORTADORA DE CÉSPED, PULVERIZADORA MANUAL, MOTOSIERRA, DESBROZADORA, MULTIHERRAMIENTA
Shipment months/Production finish date: JUNIO - 2025
Total Value: $91,100.60
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $91,100.60
Comments: AGENTE ESPERADO, TOTAL CBM - 64.9


Order No: BP-0063-II

Products: CORTADORA DE CÉSPED, PULVERIZADORA MANUAL, MOTOSIERRA, DESBROZADORA, MULTIHERRAMIENTA
Shipment months/Production finish date: JUNIO - 2025
Total Value: $72,244.40
Marketing Budget: $0.00
Payment received: $0.00
Remaining Balance: $72,244.40
Comments: AGENTE ESPERADO, TOTAL CBM - 62.23


Total

Total Value: $1,829,314.22
Marketing Budget: $11,638.46
Payment received: $0.00
Remaining Balance: $1,829,314.22


"""
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # Reduced for speed
    chunk_overlap=30
)
split_docs = text_splitter.create_documents(docs)

print(f"Total chunks created: {len(split_docs)}")
# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store flyers in Pinecone
vectorstore = PineconeVectorStore.from_texts(
    texts=docs,
    embedding=embeddings,
    index_name=index_name,
    pinecone_api_key=pinecone_api_key,
)


# Perform a test query
query = "give basic info"
results = vectorstore.similarity_search(query, k=2)

# Output the results
print("\nTop Matching Documents:\n")
for i, result in enumerate(results, start=1):
    print(f"Result {i}:\n{result.page_content}\n{'-'*60}")



