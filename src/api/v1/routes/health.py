from fastapi import APIRouter

from src.api.v1.schemas import LLMHealthResponse, ToolsHealthResponse
from src.api.v1.dependencies import LLMDep, ToolRegistryDep
from src.shared.interfaces.llm import ChatMessage, GenerationConfig, MessageRole
from src.shared.interfaces.tool import ToolContext



router = APIRouter(tags=["health"])

@router.get("/llm/health", response_model=LLMHealthResponse)
async def llm_health(llm: LLMDep):
    try:
        await llm.generate(
            messages=[
                ChatMessage(role=MessageRole.SYSTEM, content="Reply with OK only."),
                ChatMessage(role=MessageRole.USER, content="Health check"),
            ],
            config=GenerationConfig(
                temperature=0.0,
                max_tokens=5,
                timeout_s=10,
            ),
        )
        return LLMHealthResponse(
            status="ok",
            llm_ok=True,
            provider="openai",
            model=llm.model_name,
        )
    except Exception as exc:
        return LLMHealthResponse(
            status="degraded",
            llm_ok=False,
            provider="openai",
            model=llm.model_name,
            detail=str(exc),
        )


@router.get("/tools/health", response_model=ToolsHealthResponse)
async def tools_health(registry: ToolRegistryDep):
    tool_name = "ping"
    try:
        result = await registry.execute(
            tool_name=tool_name,
            arguments={"message": "health-check"},
            context=ToolContext(session_id="health-check"),
        )
        if not result.success:
            return ToolsHealthResponse(
                status="degraded",
                tools_ok=False,
                tool_name=tool_name,
                detail=result.error or "Tool execution failed.",
            )
        output = result.output if isinstance(result.output, dict) else {"value": result.output}
        return ToolsHealthResponse(
            status="ok",
            tools_ok=True,
            tool_name=tool_name,
            output=output,
        )
    except Exception as exc:
        return ToolsHealthResponse(
            status="degraded",
            tools_ok=False,
            tool_name=tool_name,
            detail=str(exc),
        )
