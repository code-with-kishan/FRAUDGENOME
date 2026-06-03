import os
import json
import time
from typing import Dict, Any, List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors


def build_brief_payload(account_id: str, cti_score: float, shap_explanations: Dict[str, float],
                        frauddna_matches: List[Dict[str, Any]], ring_summary: Optional[Dict[str, Any]] = None,
                        notes: Optional[str] = None) -> Dict[str, Any]:
    """Assemble a structured brief payload suitable for human review or LLM prompting.

    The payload is intentionally explicit so Claude or another LLM can transform it into
    human readable text. We also use the same payload to render a baseline PDF.
    """
    payload = {
        "account_id": account_id,
        "cti_score": float(cti_score),
        "shap_explanations": shap_explanations,
        "frauddna_matches": frauddna_matches,
        "ring_summary": ring_summary or {},
        "notes": notes or "",
        "generated_at": int(time.time()),
    }
    return payload


def generate_pdf_from_payload(payload: Dict[str, Any], out_path: str) -> str:
    """Generate a simple structured PDF brief from the payload.

    Returns the path to the generated PDF file.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    doc = SimpleDocTemplate(out_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    title = f"MuleGuard Brief — Account {payload.get('account_id')}"
    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 12))

    meta = f"CTI Score: {payload.get('cti_score'):.4f} — Generated: {time.ctime(payload.get('generated_at'))}"
    story.append(Paragraph(meta, styles['Normal']))
    story.append(Spacer(1, 12))

    # SHAP table
    story.append(Paragraph("SHAP Feature Contributions", styles['Heading2']))
    shap = payload.get('shap_explanations', {})
    if shap:
        rows = [["Feature", "SHAP (abs)"]]
        for k, v in sorted(shap.items(), key=lambda kv: -abs(kv[1]))[:30]:
            rows.append([k, f"{v:.5f}"])
        t = Table(rows, colWidths=[300, 100])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f2f2f2')),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No SHAP explanations available.", styles['Normal']))
    story.append(Spacer(1, 12))

    # FraudDNA / DTW matches
    story.append(Paragraph("FraudDNA / DTW Matches", styles['Heading2']))
    matches = payload.get('frauddna_matches', [])
    if matches:
        rows = [["Match Account", "Distance", "Pattern ID", "Notes"]]
        for m in matches[:20]:
            rows.append([
                str(m.get('account_id', m.get('acct', 'N/A'))),
                f"{m.get('distance', 0):.4f}",
                str(m.get('pattern_id', '')),
                m.get('note', '')[:80],
            ])
        t = Table(rows, colWidths=[150, 80, 80, 190])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f2f2f2')),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No DTW matches found.", styles['Normal']))
    story.append(Spacer(1, 12))

    # Ring summary
    story.append(Paragraph("Ring / Network Summary", styles['Heading2']))
    ring = payload.get('ring_summary', {})
    if ring:
        for k, v in ring.items():
            story.append(Paragraph(f"<b>{k}</b>: {v}", styles['Normal']))
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("No ring information available.", styles['Normal']))
    story.append(Spacer(1, 12))

    # Notes / suggested actions
    story.append(Paragraph("Suggested Actions", styles['Heading2']))
    story.append(Paragraph(payload.get('notes', ""), styles['Normal']))

    doc.build(story)
    return out_path


def build_claude_prompt_from_payload(payload: Dict[str, Any]) -> str:
    """Create a Claude-friendly prompt template containing payload and instructions.

    This is a helper for the human-in-loop flow: we produce a concise, structured prompt
    that can be sent to Claude to produce a human-polished PDF draft. The actual call
    to Claude is intentionally not implemented here — keep API call responsibility to
    the deployment layer where secure keys are stored.
    """
    prompt = [
        "You are MuleGuard AI assistant. Produce a concise forensic brief for investigators.",
        "Include: one-paragraph summary, top 5 SHAP drivers (with exact values), top 5 DTW matches (account, distance), ring summary, and recommended next actions.",
        "Output JSON with keys: short_summary, top_shap (list), top_matches (list), ring_summary, recommended_actions, citations.",
        "Payload:",
        json.dumps(payload, indent=2, default=str),
    ]
    return "\n\n".join(prompt)


if __name__ == '__main__':
    # tiny smoke-run example to generate a draft PDF
    payload = build_brief_payload(
        account_id='acct-demo',
        cti_score=0.87,
        shap_explanations={'txn_amt_mean': 0.12, 'device_age': -0.04, 'hour_of_day': 0.03},
        frauddna_matches=[{'account_id': 'mule_1', 'distance': 0.123, 'pattern_id': 'p-42', 'note': 'near-identical pattern'}],
        ring_summary={'community_id': 3, 'members': 12, 'sync_window_hours': 48},
        notes='Recommend manual review and temporary hold.'
    )
    out = os.path.join('models', 'briefs', f"brief_{payload['account_id']}.pdf")
    p = generate_pdf_from_payload(payload, out)
    print('Generated', p)
