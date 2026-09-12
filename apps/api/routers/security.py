"""API router exposing adversarial security, PII redaction, rate limiting, and red-team audits."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Header, Query, status

from apps.api.core.dependencies import require_roles
from apps.api.models.user import UserRole
from apps.api.schemas.security import (
    PIIRedactRequest,
    PIIRedactResponse,
    PromptScanRequest,
    PromptScanResponse,
    RateLimitCheckResponse,
    RateLimitResetRequest,
    RedTeamAuditRequest,
    SecurityAuditLogsResponse,
)
from security.pii_redactor import pii_redactor
from security.prompt_defense import prompt_scanner
from security.rate_limiter import rate_limiter
from security.red_team import red_team_engine

router = APIRouter(prefix="/security", tags=["Security & Guardrails"])


@router.post("/scan-prompt", response_model=PromptScanResponse, summary="Scan prompt for adversarial injection")
async def scan_prompt(request: PromptScanRequest) -> PromptScanResponse:
    """Analyze input prompt for injection, jailbreaks, role reversal, and encoding evasion."""
    result = prompt_scanner.scan(request.prompt)
    return PromptScanResponse(
        is_safe=result.is_safe,
        action=result.action,
        severity=result.severity,
        threat_score=result.threat_score,
        detected_threats=result.detected_threats,
        matched_rules=result.matched_rules,
        sanitized_prompt=result.sanitized_prompt,
        scan_time_ms=result.scan_time_ms,
    )


@router.post("/redact-pii", response_model=PIIRedactResponse, summary="Scan and redact sensitive PII and secrets")
async def redact_pii(request: PIIRedactRequest) -> PIIRedactResponse:
    """Mask credit cards (Luhn-checked), SSNs, emails, phone numbers, and cloud API keys."""
    result = pii_redactor.scan_and_redact(request.text)
    return PIIRedactResponse(
        contains_pii=result.contains_pii,
        findings_count=result.findings_count,
        findings=result.findings,
        redacted_text=result.redacted_text,
    )


@router.get("/rate-limit/status", response_model=RateLimitCheckResponse, summary="Check rate limit status")
async def get_rate_limit_status(
    client_id: str = Query(..., description="Client identifier or API token"),
    tier: str = Query("free", description="Client subscription tier"),
) -> RateLimitCheckResponse:
    """Inspect client token-bucket rate limit quota and remaining requests."""
    status_res = rate_limiter.peek(client_id, tier=tier)
    return RateLimitCheckResponse(status=status_res)


@router.post(
    "/rate-limit/reset",
    status_code=status.HTTP_200_OK,
    summary="Reset rate limit quota for client (Admin only)",
)
async def reset_rate_limit(
    request: RateLimitResetRequest,
    current_admin=Depends(require_roles(UserRole.ADMIN)),
) -> dict:
    """Admin endpoint to reset rate limiter tokens for a client."""
    rate_limiter.reset_client(request.client_id)
    return {"status": "success", "message": f"Rate limit reset for {request.client_id}"}


@router.post(
    "/red-team/audit",
    response_model=SecurityAuditLogsResponse,
    summary="Execute automated adversarial red-team audit battery",
)
async def run_red_team_audit(
    request: Optional[RedTeamAuditRequest] = None,
    x_admin_token: Optional[str] = Header(None),
) -> SecurityAuditLogsResponse:
    """Run 32-vector adversarial attack battery and generate OWASP LLM Top 10 compliance score."""
    report = red_team_engine.run_audit()

    events = [
        {
            "attack_id": r.attack_id,
            "name": r.attack_name,
            "type": r.attack_type.value,
            "blocked": r.blocked,
            "threat_score": r.threat_score,
            "action": r.action_taken.value,
        }
        for r in report.results
    ]

    return SecurityAuditLogsResponse(
        events_count=len(events),
        events=events,
        report=report,
    )
