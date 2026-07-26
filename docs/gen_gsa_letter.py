"""GSA Capital cover letter — leads with research methodology."""
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

OUT = "/home/user/David/docs/Belyn_GSA_Capital_CoverLetter.pdf"
INK = HexColor("#111111")

name_s = ParagraphStyle("n", fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=INK, spaceAfter=1)
contact_s = ParagraphStyle("c", fontName="Helvetica", fontSize=9.2, leading=12, textColor=INK, spaceAfter=13)
plain_s = ParagraphStyle("p", fontName="Helvetica", fontSize=9.4, leading=12.4, textColor=INK, spaceAfter=7)
body_s = ParagraphStyle("b", fontName="Helvetica", fontSize=9.4, leading=12.5, textColor=INK,
                        alignment=TA_JUSTIFY, spaceAfter=7)

S = [
    Paragraph("DAVID N. BELYN", name_s),
    Paragraph("(717) 437-3885 &nbsp;|&nbsp; dbelyn@hotmail.com", contact_s),
    Paragraph("July 25, 2026", plain_s),
    Paragraph("Research Recruitment<br/>GSA Capital", plain_s),
    Paragraph("<b>Re: Quantitative Researcher</b>", plain_s),
    Paragraph("To the GSA Capital research team,", plain_s),

    Paragraph(
        "My route to quantitative research is unconventional, so I will lead with the work rather than "
        "the CV.", body_s),

    Paragraph(
        "This year I designed, built, and ran a systematic trading platform end to end, and used it to "
        "test whether a series of candidate FX strategies carried any edge. The infrastructure was "
        "production-grade — autonomous decision agent, deterministic server-side risk gates, live "
        "broker integration, streaming data, real-time monitoring — but the methodology is the part "
        "worth your time, not the plumbing.", body_s),

    Paragraph(
        "I pre-registered acceptance criteria before examining results, held data out strictly for "
        "verification, and reported accuracy with Wilson confidence bounds. Four hypotheses went through "
        "it: an intraday directional model evaluated across 2,500+ logged forecasts; a five-factor, "
        "conviction-gated ensemble (n = 325 out-of-sample, 49.8%, lower bound 0.453); time-series carry "
        "and trend across eight majors over 14.9 years; and cross-sectional carry across sixteen pairs "
        "including the JPY funding crosses. The two backtested families returned Sharpe 0.00 and 0.14 "
        "against a standard error of roughly ±0.26. A data-mined conditional signal scoring 68.9% "
        "in-sample scored 31.3% out-of-sample, with individual cells inverting rather than decaying — "
        "the signature of fitting time-correlated noise rather than structure.", body_s),

    Paragraph(
        "Before trusting any of that, I validated the instrument itself. The same engine — lookahead-safe "
        "by construction, transaction costs charged on turnover, financing accrued daily, and no fitted "
        "parameters — recovered the equity risk premium from index CFDs at Sharpe 0.55 net of costs. The "
        "nulls were measurements rather than malfunctions. I had earlier validated the engine against "
        "synthetic series with known properties: trending, random-walk, and drawdown regimes.", body_s),

    Paragraph(
        "A negative result is an unusual thing to lead a letter with. I do it deliberately: the discipline "
        "required to retire four of one's own hypotheses, and to build the positive control that makes "
        "retiring them credible, is the substance of the work rather than an aside to it — and the "
        "difference between a robust trading algorithm and an overfitted one.", body_s),

    Paragraph(
        "My formal credentials are a B.S. in Finance (Auburn, magna cum laude) and dual master's degrees "
        "in Applied Economics and Applied Analytics at Boston College (3.87 GPA, expected 2027), with "
        "graduate coursework in econometrics, machine learning, and statistical modelling. My "
        "professional background is twenty-three years as a U.S. Army officer, currently an information "
        "operations planner. That is not quantitative finance. It is two decades of forming and defending "
        "conclusions under genuine uncertainty, before audiences unmoved by confident "
        "presentation.", body_s),

    Paragraph(
        "On timing: I complete both master's degrees in spring 2027 and retire from the Army that "
        "summer, with availability to begin in August 2027. I am willing to relocate to London. If that "
        "timeline fits a graduate research intake rather than an experienced-hire process, I would be "
        "glad to be considered in that cohort.", body_s),

    Paragraph(
        "I recognise this is not the profile you typically see, and I would not ask you to take the "
        "methodology on faith: the platform, the code, and every result above are public and "
        "reproducible from free data with a single command. If the work suggests someone worth half "
        "an hour, I would welcome the conversation.", body_s),

    Spacer(1, 6),
    Paragraph("Respectfully,", plain_s),
    Spacer(1, 10),
    Paragraph("David N. Belyn", plain_s),
]

SimpleDocTemplate(OUT, pagesize=LETTER, leftMargin=0.85*inch, rightMargin=0.85*inch,
                  topMargin=0.6*inch, bottomMargin=0.5*inch,
                  title="David N. Belyn — Cover Letter", author="David N. Belyn").build(S)
print("wrote", OUT)
