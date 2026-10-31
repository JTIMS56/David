"""Generate David Belyn's updated quantitative-finance resume (with Popper)."""
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
                                KeepTogether)

OUT = "/home/user/David/docs/Belyn_Quant_Resume.pdf"
INK = HexColor("#111111")
RULE = HexColor("#666666")

name_s = ParagraphStyle("name", fontName="Helvetica-Bold", fontSize=17, leading=20,
                        textColor=INK, spaceAfter=2)
contact_s = ParagraphStyle("contact", fontName="Helvetica", fontSize=8.8, leading=11,
                           textColor=INK, spaceAfter=8)
sec_s = ParagraphStyle("sec", fontName="Helvetica-Bold", fontSize=9.6, leading=12,
                       textColor=INK, spaceBefore=8, spaceAfter=1)
body_s = ParagraphStyle("body", fontName="Helvetica", fontSize=8.9, leading=11.4,
                        textColor=INK, alignment=TA_JUSTIFY, spaceAfter=3)
comp_s = ParagraphStyle("comp", fontName="Helvetica", fontSize=8.7, leading=11.6,
                        textColor=INK, spaceAfter=2)
role_s = ParagraphStyle("role", fontName="Helvetica-Bold", fontSize=9.1, leading=11.6,
                        textColor=INK, spaceBefore=4, spaceAfter=1)
bullet_s = ParagraphStyle("bullet", fontName="Helvetica", fontSize=8.7, leading=11.0,
                          textColor=INK, leftIndent=11, bulletIndent=2,
                          alignment=TA_JUSTIFY, spaceAfter=1.6)

def section(title):
    return [Paragraph(title, sec_s),
            HRFlowable(width="100%", thickness=0.7, color=RULE,
                       spaceBefore=1, spaceAfter=4)]

def role(title, right):
    """Role line with right-aligned date via a tab-stop table-free trick."""
    return Paragraph(
        f'<para>{title}<font color="#444444"> &nbsp;|&nbsp; {right}</font></para>', role_s)

def b(text):
    return Paragraph(text, bullet_s, bulletText="•")

story = []

# ── Header ────────────────────────────────────────────────────────────────────
story.append(Paragraph("DAVID N. BELYN", name_s))
story.append(Paragraph(
    "Richmond Hill, GA &nbsp;•&nbsp; (717) 437-3885 &nbsp;•&nbsp; dbelyn@hotmail.com "
    "&nbsp;•&nbsp; Top Secret/SCI Clearance", contact_s))

# ── Summary ───────────────────────────────────────────────────────────────────
story += section("PROFESSIONAL SUMMARY")
story.append(Paragraph(
    "Analyst and strategist with 23+ years translating complex, ambiguous environments into "
    "actionable insight and quantitative decision support. Expertise spanning statistical modeling, "
    "risk assessment, strategic planning, and information operations across 15+ countries. Currently "
    "pursuing dual master's degrees in Applied Analytics and Applied Economics at Boston College "
    "(3.87 GPA), with graduate coursework in machine learning, econometrics, and statistical modeling. "
    "Recently designed and deployed an autonomous algorithmic trading and strategy-validation platform "
    "on live broker infrastructure, and published its out-of-sample findings. Combines quantitative "
    "rigor, hands-on Python, C++, and SQL engineering, and a record of disciplined judgment under "
    "uncertainty. Seeking to apply this background to quantitative modeling, analytics, and risk "
    "management within a financial institution.", body_s))

# ── Core competencies ─────────────────────────────────────────────────────────
story += section("CORE COMPETENCIES")
story.append(Paragraph(
    "Quantitative &amp; Statistical Analysis &nbsp;•&nbsp; Econometric &amp; Predictive Modeling "
    "&nbsp;•&nbsp; Backtesting &amp; Out-of-Sample Validation &nbsp;•&nbsp; Algorithmic Trading Systems "
    "&nbsp;•&nbsp; Risk Assessment &amp; Decision Support &nbsp;•&nbsp; Market &amp; Operational Risk "
    "Concepts &nbsp;•&nbsp; Portfolio &amp; Resource Management &nbsp;•&nbsp; Python, R, SQL, ML "
    "Pipelines &nbsp;•&nbsp; Data Visualization &amp; Reporting &nbsp;•&nbsp; Executive Briefing "
    "&amp; Presentations", comp_s))

# ── Education ─────────────────────────────────────────────────────────────────
story += section("EDUCATION")
story.append(role("M.S. Applied Analytics &amp; M.S. Applied Economics, Boston College",
                  "Expected Spring 2027"))
story.append(b("Woods College of Advancing Studies. GPA 3.87. Coursework: machine learning, NLP, "
               "econometrics, statistical modeling, data visualization, game theory, industrial "
               "organization, market dynamics."))
story.append(role("B.S. Business Administration, Finance, Auburn University", "May 2011"))
story.append(b("Magna Cum Laude with Honors."))

# ── Projects ──────────────────────────────────────────────────────────────────
story += section("QUANTITATIVE &amp; RESEARCH PROJECTS")

story.append(role("POPPER — Autonomous Trading &amp; Strategy Validation Platform", "2026"))
story.append(b("Designed and deployed a production algorithmic trading system against live broker "
               "infrastructure (OANDA v20): autonomous decision agent, deterministic server-side risk "
               "controls, streaming market data, and a real-time monitoring dashboard. Python, FastAPI, "
               "PostgreSQL, Docker."))
story.append(b("Built a strategy-validation pipeline using pre-registered acceptance criteria, "
               "out-of-sample splits, and Wilson confidence intervals; correctly rejected four candidate "
               "strategy families, including a data-mined signal that scored 77% in-sample and 15% "
               "out-of-sample."))
story.append(b("Implemented a lookahead-safe backtesting engine with transaction costs and financing "
               "accrual across 14+ years of daily data; verified engine integrity with a positive control "
               "that recovered the equity risk premium at Sharpe 0.55 net of costs."))
story.append(b("Engineered graduated execution tiers and hard pre-trade risk gates (exposure caps, "
               "volatility floors, event blackouts, daily-loss halt, kill switch) that contained two "
               "production defects, including a multi-container state-divergence failure, to under $2 "
               "in realized cost."))
story.append(b("Published the methodology and negative results; codebase, incident analysis, and "
               "findings documented publicly."))

story.append(role("MERIDIAN — Global Media Intelligence Dashboard", "2024–Present"))
story.append(b("Built a full-stack Flask application using GDELT data to monitor global trends and "
               "narrative patterns across 100+ countries, deployed on Render."))
story.append(b("Applied BERTopic, transformer-based sentiment analysis, and custom frame scoring to "
               "surface emerging signals across sectors including fintech, defense technology, and "
               "logistics; methodology targeted for publication in <i>Parameters</i>."))

story.append(role("Work-From-Home Equity &amp; Labor Market Research", "2025"))
story.append(b("Conducted primary analysis of IPUMS CPS microdata on remote work and employment equity; "
               "presented as a competitive poster at the BC Analytics &amp; Industry Symposium. Findings "
               "challenged prevailing assumptions, demonstrating the ability to interrogate market "
               "narratives with data."))

story.append(role("Capital One Case Competition — International Expansion Analysis", "2024"))
story.append(b("Built market-entry recommendations for Vietnam and Brazil, synthesizing macroeconomic "
               "data, competitive dynamics, and risk considerations into executive-ready deliverables."))

# ── Experience ────────────────────────────────────────────────────────────────
story += section("PROFESSIONAL EXPERIENCE")

story.append(role("G39 Planner, Information Operations, 3rd Infantry Division", "2024–Present"))
story.append(b("Lead strategic planning for a 20,000-person division, synthesizing intelligence on "
               "adversary messaging, information environment dynamics, and emerging technology risk."))
story.append(b("Oversee execution of a ~$25M annual contracted maintenance program; track obligations "
               "and burn rates against approved funding and present funding recommendations to the "
               "Commanding General."))
story.append(b("Develop analytical decision-support products for general officer audiences, translating "
               "technical and operational complexity into clear frameworks."))

story.append(role("Operations Advisor to Chief of General Staff, U.S. Military Training Mission, "
                  "Saudi Arabia", "2023–2024"))
story.append(b("Senior strategic advisor at the ministerial level for one of the largest U.S. defense "
               "partnerships, advising on modernization across a $20B+ annual budget environment."))
story.append(b("Assessed risk and resource tradeoffs in a high-stakes, high-ambiguity environment, "
               "producing analysis consumed at the highest levels of government."))

story.append(role("Battalion Executive Officer, 2nd Battalion, 47th Infantry Regiment", "2022–2023"))
story.append(b("Second-in-command of a 600-person organization; managed budget execution, $2.1M+ in "
               "equipment assets, and cross-functional staff operations."))

story.append(role("Special Mission Unit Operational Planner, Special Operations Command Central",
                  "2021–2022"))
story.append(b("Deployed as forward planner; developed operational plans and provided real-time "
               "decision support to senior command under significant uncertainty."))

story.append(role("Chief of Plans (S3), 198th Infantry Brigade", "2018–2020"))
story.append(b("Directed brigade-level strategic planning; produced campaign plans aligned to Joint "
               "Chiefs and USSOUTHCOM direction and delivered analytical products for senior leadership."))

story.append(role("Company Commander, 3-60th Infantry Regiment", "2015–2016"))
story.append(b("Commanded a 240-soldier training company with full accountability for $2.1M in "
               "equipment, sustaining a 98% completion rate across a 1,000+ person annual mission."))

# ── Technical skills ──────────────────────────────────────────────────────────
story += section("TECHNICAL SKILLS")
story.append(Paragraph(
    "<b>Languages &amp; Frameworks:</b> Python (Pandas, NumPy, Scikit-learn, FastAPI, Flask, ML/NLP "
    "pipelines), R, SQL, C++, React/JavaScript", comp_s))
story.append(Paragraph(
    "<b>Modeling &amp; Statistics:</b> Econometric modeling, regression and classification (logistic "
    "regression, Linear SVM), time-series backtesting, hypothesis testing and confidence intervals, "
    "topic modeling (LDA/NMF, BERTopic), transformer models", comp_s))
story.append(Paragraph(
    "<b>Systems &amp; Data:</b> PostgreSQL, SQLAlchemy, Docker, REST and WebSocket APIs, Anthropic API "
    "(agentic tool use), OANDA v20, GDELT, IPUMS CPS", comp_s))
story.append(Paragraph(
    "<b>Analytics &amp; Visualization:</b> Tableau, ggplot2, matplotlib, Excel (advanced)", comp_s))

# ── Honors ────────────────────────────────────────────────────────────────────
story += section("HONORS &amp; RECOGNITION")
story.append(Paragraph(
    "Bronze Star Medal &nbsp;•&nbsp; Meritorious Service Medal &nbsp;•&nbsp; Joint Service Commendation "
    "Medal &nbsp;•&nbsp; Army Commendation Medal (5x) &nbsp;•&nbsp; Combat Infantryman's Badge "
    "&nbsp;•&nbsp; Expert Infantryman's Badge &nbsp;•&nbsp; Edward R. Murrow Award for Continuing "
    "Coverage (2002), Broadcast Journalism", comp_s))

def later_pages(canvas, doc_):
    """Running header on pages 2+ so a separated page still identifies itself."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(HexColor("#555555"))
    canvas.drawString(0.62*inch, LETTER[1] - 0.34*inch, "David N. Belyn")
    canvas.drawRightString(LETTER[0] - 0.62*inch, LETTER[1] - 0.34*inch,
                           f"Page {doc_.page}")
    canvas.restoreState()

doc = SimpleDocTemplate(OUT, pagesize=LETTER,
                        leftMargin=0.62*inch, rightMargin=0.62*inch,
                        topMargin=0.5*inch, bottomMargin=0.45*inch,
                        title="David N. Belyn — Resume", author="David N. Belyn")
doc.build(story, onLaterPages=later_pages)
print("wrote", OUT)
