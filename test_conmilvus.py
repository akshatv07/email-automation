from pymilvus import connections
import config.settings as settings

connections.connect(
    alias="default",
    host=settings.MILVUS_HOST,
    port=settings.MILVUS_PORT
)

print("✅ Connected to Milvus")
