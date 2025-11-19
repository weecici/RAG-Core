import inngest
import src.services.public as public_svcs
from fastapi import APIRouter
from src import schemas
from src.core import inngest_client

router = APIRouter()


@inngest_client.create_function(
    fn_id="ingest-documents",
    trigger=inngest.TriggerEvent(event="rag/ingest-documents"),
    retries=0,
)
async def ingest_documents(ctx: inngest.Context) -> dict[str, any]:
    request = schemas.IngestionRequest.model_validate(ctx.event.data)
    return public_svcs.ingest_documents(request).model_dump()


@router.post(
    "/ingest/documents",
    response_model=schemas.IngestionResponse,
    summary="Start document ingestion",
    description="Ingest documents from the specified file paths or directory.",
)
async def ingest_documents2(
    request: schemas.IngestionRequest,
) -> schemas.IngestionResponse:
    return public_svcs.ingest_documents(request)


@inngest_client.create_function(
    fn_id="ingest-ir-dataset",
    trigger=inngest.TriggerEvent(event="rag/ingest-ir-dataset"),
    retries=0,
)
async def ingest_ir_dataset(ctx: inngest.Context) -> dict[str, any]:
    request = schemas.IRDatasetIngestionRequest.model_validate(ctx.event.data)
    return public_svcs.ingest_ir_dataset(request).model_dump()


@router.post(
    "/ingest/ir-dataset",
    response_model=schemas.IngestionResponse,
    summary="IR dataset ingestion",
    description="Ingest IR dataset",
)
async def ingest_ir_dataset2(
    request: schemas.IRDatasetIngestionRequest,
) -> schemas.IngestionResponse:
    return public_svcs.ingest_ir_dataset(request)
