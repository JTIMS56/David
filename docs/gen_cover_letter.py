"""F_Trad_Fi cover letter — honest version (no claimed live edge or track record)."""
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

OUT = "/home/user/David/docs/Belyn_F_Trad_Fi_CoverLetter.pdf"
INK = HexColor("#111111")

name_s = ParagraphStyle("n", fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=INK, spaceAfter=1)
contact_s = ParagraphStyle("c", fontName="Helvetica", fontSize=9.2, leading=12, textColor=INK, spaceAfter=14)
plain_s = ParagraphStyle("p", fontName="Helvetica", fontSize=9.6, leading=13, textColor=INK, spaceAfter=9)
body_s = ParagraphStyle("b", fontName="Helvetica", fontSize=9.6, leading=13.6, textColor=INK,
                        alignment=TA_JUSTIFY, spaceAfter=9)

S = [
    Paragraph("DAVID N. BELYN", name_s),
    Paragraph("(717) 437-3885 &nbsp;|&nbsp; dbelyn@hotmail.com", contact_s),
    Paragraph("July 25, 2026", plain_s),
    Paragraph("Hiring Team<br/>F_Trad_Fi", plain_s),
    Paragraph("<b>Re: Quantitative Trader — Prediction Markets</b>", plain_s),
    Paragraph("To the F_Trad_Fi team,", plain_s),

    Paragraph(
        "I will be as direct as your posting is. You asked for a verifiable live track record and a "
        "demonstrated edge. I do not have one in prediction markets. I am writing anyway, briefly, so "
        "you can decide in thirty seconds whether the rest is worth your time.", body_s),

    Paragraph(
        "This summer I built and deployed an autonomous systematic trading platform end to end: an "
        "LLM decision agent, deterministic server-side risk gates, live OANDA v20 integration, "
        "streaming market data, PostgreSQL, and real-time monitoring, running against a practice "
        "account while I tested whether the strategies driving it had any edge at all.", body_s),

    Paragraph(
        "They did not. I ran four pre-registered, out-of-sample tests spanning intraday signal models, "
        "a five-factor ensemble, and both time-series and cross-sectional carry across fourteen years "
        "of daily data. Every result came back statistically indistinguishable from zero. A conditional "
        "signal that scored 77% in-sample scored 15% out-of-sample. I published the finding rather than "
        "the fantasy.", body_s),

    Paragraph(
        "What I think earns your attention is the next step. Before trusting those nulls I validated the "
        "instrument itself: the same backtesting engine, costs and financing charged, pointed at equity "
        "indices, recovered the equity risk premium at Sharpe 0.55. The zeros were readings, not "
        "malfunctions. Along the way I caught two production defects, including a multi-container "
        "state-divergence failure that was manufacturing phantom fills and quietly corrupting my own "
        "evaluation data, because I had architected graduated execution tiers that capped the cost of "
        "being wrong at under two dollars.", body_s),

    Paragraph(
        "That is most of the list you published under a different name: execution quality, failure "
        "modes, drawdown discipline, and edge durability. What I do not have is six months of realized "
        "P&amp;L on Polymarket or Kalshi, and I will not dress up a practice account as one.", body_s),

    Paragraph(
        "So, plainly: if you need someone who already runs a profitable prediction-markets book, I am "
        "not that person today, and I would rather tell you now than waste an interview slot. If there "
        "is value in someone who builds the research, validation, and execution infrastructure such a "
        "book runs on, who has demonstrated he will kill his own thesis when the data says so, and who "
        "is working toward a live book of his own, I would welcome a conversation on those terms.", body_s),

    Paragraph(
        "My background is twenty-three years as a U.S. Army officer, currently an information operations "
        "planner for the 3rd Infantry Division, finishing dual master's degrees in applied economics and "
        "applied analytics at Boston College. I have spent a career making decisions on incomplete "
        "information and defending them to people who ask hard questions. The platform, the code, the "
        "incident write-ups, and the findings are all public, and I am glad to walk through any part of "
        "them.", body_s),

    Spacer(1, 6),
    Paragraph("Respectfully,", plain_s),
    Spacer(1, 10),
    Paragraph("David N. Belyn", plain_s),
]

SimpleDocTemplate(OUT, pagesize=LETTER, leftMargin=0.9*inch, rightMargin=0.9*inch,
                  topMargin=0.75*inch, bottomMargin=0.6*inch,
                  title="David N. Belyn — Cover Letter", author="David N. Belyn").build(S)
print("wrote", OUT)
