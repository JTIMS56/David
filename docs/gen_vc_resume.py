"""VC/tech-targeted resume variant featuring Popper."""
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, HRFlowable

OUT = "/home/user/David/docs/Belyn_VC_Tech_Resume.pdf"
INK, RULE = HexColor("#111111"), HexColor("#666666")

name_s = ParagraphStyle("n", fontName="Helvetica-Bold", fontSize=17, leading=20, textColor=INK, spaceAfter=2)
contact_s = ParagraphStyle("c", fontName="Helvetica", fontSize=8.8, leading=11, textColor=INK, spaceAfter=8)
sec_s = ParagraphStyle("s", fontName="Helvetica-Bold", fontSize=9.6, leading=12, textColor=INK, spaceBefore=8, spaceAfter=1)
body_s = ParagraphStyle("b", fontName="Helvetica", fontSize=8.9, leading=11.4, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=3)
comp_s = ParagraphStyle("cp", fontName="Helvetica", fontSize=8.7, leading=11.6, textColor=INK, spaceAfter=2)
role_s = ParagraphStyle("r", fontName="Helvetica-Bold", fontSize=9.1, leading=11.6, textColor=INK, spaceBefore=4, spaceAfter=1)
bul_s = ParagraphStyle("bu", fontName="Helvetica", fontSize=8.7, leading=11.0, textColor=INK,
                       leftIndent=11, bulletIndent=2, alignment=TA_JUSTIFY, spaceAfter=1.6)

def sec(t): return [Paragraph(t, sec_s), HRFlowable(width="100%", thickness=0.7, color=RULE, spaceBefore=1, spaceAfter=4)]
def role(t, d): return Paragraph(f'<para>{t}<font color="#444444"> &nbsp;|&nbsp; {d}</font></para>', role_s)
def b(t): return Paragraph(t, bul_s, bulletText="•")

S = []
S.append(Paragraph("DAVID N. BELYN", name_s))
S.append(Paragraph("Richmond Hill, GA &nbsp;•&nbsp; (717) 437-3885 &nbsp;•&nbsp; dbelyn@hotmail.com "
                   "&nbsp;•&nbsp; Top Secret/SCI Clearance", contact_s))

S += sec("PROFESSIONAL SUMMARY")
S.append(Paragraph(
    "Senior strategist and analyst with 23+ years translating complex, ambiguous environments into "
    "actionable intelligence and investment-quality insight. Expertise spanning technology assessment, "
    "information operations, market dynamics, and multi-domain strategic planning across 15+ countries. "
    "Currently pursuing dual master's degrees (Applied Analytics and Applied Economics, Boston College, "
    "3.87 GPA) with graduate-level AI/ML coursework. Builds as well as evaluates: recently shipped and "
    "deployed an autonomous LLM-agent trading platform end-to-end, then published its out-of-sample "
    "findings, including the evidence that killed the original thesis. Brings a rare combination of "
    "hands-on agentic AI engineering, quantitative rigor, and credibility assessing technology at "
    "scale. Eager to apply this background to evaluating startups and supporting investment decisions "
    "at an early-stage firm.", body_s))

S += sec("CORE COMPETENCIES")
S.append(Paragraph(
    "Technology Sector Analysis &nbsp;•&nbsp; AI/ML Research &amp; Applied Engineering &nbsp;•&nbsp; "
    "Agentic System Design &nbsp;•&nbsp; Competitive Intelligence &nbsp;•&nbsp; Market Research &amp; "
    "Synthesis &nbsp;•&nbsp; Investment Memos &amp; Diligence Frameworks &nbsp;•&nbsp; Quantitative "
    "&amp; Statistical Analysis &nbsp;•&nbsp; Python, R, SQL, ML Pipelines &nbsp;•&nbsp; Executive "
    "Briefing &nbsp;•&nbsp; Startup Ecosystem &amp; Defense Tech", comp_s))

S += sec("EDUCATION")
S.append(role("M.S. Applied Analytics &amp; M.S. Applied Economics, Boston College", "Expected Spring 2027"))
S.append(b("Woods College of Advancing Studies. GPA 3.87. Coursework: machine learning, NLP, AI/ML, "
           "econometrics, statistical modeling, game theory, industrial organization, market dynamics."))
S.append(role("B.S. Business Administration, Finance, Auburn University", "May 2011"))
S.append(b("Magna Cum Laude with Honors."))

S += sec("TECHNICAL &amp; RESEARCH PROJECTS")

S.append(role("POPPER — Autonomous AI Trading &amp; Strategy Validation Platform", "2026"))
S.append(b("Shipped a production autonomous trading system end-to-end: LLM decision agent (Anthropic "
           "tool-use loop), deterministic server-side risk controls, live broker API integration, "
           "streaming market data, and a real-time dashboard. Python, FastAPI, PostgreSQL, Docker, "
           "deployed on cloud infrastructure."))
S.append(b("Designed the validation methodology that governed it — pre-registered acceptance criteria, "
           "out-of-sample testing, confidence bounds — then followed the evidence to a negative result, "
           "retiring four strategy families including one that scored 77% in-sample and 15% "
           "out-of-sample."))
S.append(b("Built a lookahead-safe backtesting engine spanning 14+ years of daily data with transaction "
           "costs and financing modeled; validated the instrument itself with a positive control that "
           "recovered the equity risk premium at Sharpe 0.55."))
S.append(b("Practical command of agentic systems in production: tool schemas and structured outputs, "
           "guardrails that hold when the model errs, cost and latency tradeoffs, and the distance "
           "between an agent demo and deployed infrastructure."))
S.append(b("Published the methodology and findings; codebase and incident analysis public on GitHub."))

S.append(role("MERIDIAN — Global Media Intelligence Dashboard", "2024–Present"))
S.append(b("Built a full-stack Flask application using GDELT data to monitor global news trends and "
           "narrative patterns across 100+ countries, deployed on Render."))
S.append(b("Applied BERTopic, transformer-based sentiment analysis, and custom frame scoring to surface "
           "emerging narratives relevant to investment themes (AI, fintech, defense tech, logistics); "
           "methodology targeted for publication in <i>Parameters</i>."))

S.append(role("WFH Equity &amp; Labor Market Research — BC Analytics &amp; Industry Symposium", "2025"))
S.append(b("Primary analysis of IPUMS CPS microdata on remote work and employment equity, presented as a "
           "competitive poster; findings challenged prevailing assumptions, demonstrating the ability to "
           "interrogate market narratives with data."))

S.append(role("Capital One Case Competition — International Expansion Analysis", "2024"))
S.append(b("Built investment-style market-entry recommendations for Vietnam and Brazil, synthesizing "
           "macroeconomic data, competitive dynamics, and go-to-market constraints into executive-ready "
           "deliverables."))

S += sec("PROFESSIONAL EXPERIENCE")
S.append(role("G39 Planner, Information Operations, 3rd Infantry Division", "2024–Present"))
S.append(b("Lead strategic IO planning for a 20,000-person division, synthesizing intelligence on "
           "adversary messaging, information environment dynamics, and emerging technology threats."))
S.append(b("Develop investment-grade analytical products for general officer audiences, translating "
           "technical and operational complexity into clear decision-support frameworks."))

S.append(role("Operations Advisor to Chief of General Staff, U.S. Military Training Mission, Saudi Arabia",
              "2023–2024"))
S.append(b("Senior strategic advisor at the ministerial level for one of the largest U.S. defense "
           "partnerships, advising on modernization across a $20B+ annual budget environment."))
S.append(b("Direct exposure to defense-tech procurement, autonomous systems evaluation, and the "
           "intersection of commercial technology with national security requirements."))

S.append(role("Battalion Executive Officer, 2nd Bn, 47th Infantry Regiment", "2022–2023"))
S.append(b("Second-in-command of a 600-person organization; managed budget execution, $2.1M+ in equipment "
           "assets, and cross-functional staff operations."))

S.append(role("Special Mission Unit Operational Planner, Special Operations Command Central", "2021–2022"))
S.append(b("Deployed as forward planner; developed operational plans and provided real-time decision "
           "support to senior command in a high-ambiguity environment."))

S.append(role("Chief of Plans (S3), 198th Infantry Brigade", "2018–2020"))
S.append(b("Directed brigade-level strategic planning; produced campaign plans aligned to Joint Chiefs and "
           "USSOUTHCOM direction and delivered analytical products for senior leadership."))

S.append(role("Chief of Reconnaissance / Training Developer, Maneuver Center of Excellence", "2012–2013"))
S.append(b("Directed research, writing, and publication of six doctrine manuals; led a 30-person "
           "cross-functional team producing content consumed Army-wide — long-form technical publishing "
           "under institutional review."))

S += sec("TECHNICAL SKILLS")
S.append(Paragraph("<b>Languages &amp; Frameworks:</b> Python (Pandas, NumPy, Scikit-learn, FastAPI, Flask, "
                   "ML/NLP pipelines), R, SQL, C++, React/JavaScript", comp_s))
S.append(Paragraph("<b>AI/ML:</b> Anthropic API (agentic tool use, function calling, structured outputs), "
                   "transformer models (HuggingFace), BERTopic, BiLSTM, TF-IDF pipelines, LDA/NMF topic "
                   "modeling, model provisioning", comp_s))
S.append(Paragraph("<b>Systems &amp; Data:</b> PostgreSQL, SQLAlchemy, Docker, REST and WebSocket APIs, "
                   "cloud deployment (DigitalOcean, Render), GDELT, IPUMS CPS", comp_s))
S.append(Paragraph("<b>Analytics:</b> Econometric modeling, classification (Linear SVM, logistic "
                   "regression), time-series backtesting, hypothesis testing, Tableau, ggplot2, "
                   "matplotlib, Excel", comp_s))

S += sec("HONORS &amp; RECOGNITION")
S.append(Paragraph("Bronze Star Medal &nbsp;•&nbsp; Meritorious Service Medal &nbsp;•&nbsp; Joint Service "
                   "Commendation Medal &nbsp;•&nbsp; Army Commendation Medal (5x) &nbsp;•&nbsp; Combat "
                   "Infantryman's Badge &nbsp;•&nbsp; Expert Infantryman's Badge &nbsp;•&nbsp; Edward R. "
                   "Murrow Award for Continuing Coverage (2002), Broadcast Journalism", comp_s))

def later(c, d):
    c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(HexColor("#555555"))
    c.drawString(0.62*inch, LETTER[1]-0.34*inch, "David N. Belyn")
    c.drawRightString(LETTER[0]-0.62*inch, LETTER[1]-0.34*inch, f"Page {d.page}")
    c.restoreState()

SimpleDocTemplate(OUT, pagesize=LETTER, leftMargin=0.62*inch, rightMargin=0.62*inch,
                  topMargin=0.5*inch, bottomMargin=0.45*inch,
                  title="David N. Belyn — Resume", author="David N. Belyn"
                  ).build(S, onLaterPages=later)
print("wrote", OUT)
