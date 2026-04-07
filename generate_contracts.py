"""
RAG Dataset Contract Generator — 50 Procurement Contracts
Generates a diverse set of synthetic legal contracts for testing the RAG pipeline.
"""

import os
import json
import time
from dotenv import load_dotenv
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from google import genai
from google.genai import types

# -----------------------------
# Config
# -----------------------------
OUTPUT_DIR = "dummy_dataset"
PDF_DIR = os.path.join(OUTPUT_DIR, "pdfs")
META_DIR = os.path.join(OUTPUT_DIR, "metadata")
DELAY_BETWEEN_CONTRACTS = 5  # seconds — avoids Gemini free-tier rate limits

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

# -----------------------------
# Load API Key & Init Client
# -----------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)
MODEL_ID = "gemini-2.5-flash"

# -----------------------------
# PDF Renderer
# -----------------------------
class ContractPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, "CONFIDENTIAL & PROPRIETARY", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def save_pdf(text, filepath):
    pdf = ContractPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=11)
    sanitized = text.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 5, sanitized)
    pdf.output(filepath)


# -----------------------------
# Gemini Generation
# -----------------------------
@retry(wait=wait_exponential(multiplier=2, min=5, max=60), stop=stop_after_attempt(7))
def generate_contract(prompt):
    system_prompt = (
        "You are an expert legal drafting AI generating synthetic legal contracts for testing datasets. "
        "CRITICAL RULES: "
        "1. NEVER use placeholders like [Client Name], [Date], [Number], etc. "
        "2. ALWAYS generate realistic but fictional data: company names, addresses, dates, prices, currencies, percentages, durations. "
        "3. Fill every field with believable values. "
        "4. OUTPUT PURE PLAIN TEXT ONLY. NO MARKDOWN. "
        "5. USE ONLY ASCII CHARACTERS to ensure clean PDF rendering. "
        "6. Use proper section numbering (1, 1.1, 1.2, 2, etc.) and clear headings."
    )
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
        ),
    )
    return response.text


# -----------------------------
# Chunking for metadata JSON
# -----------------------------
def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# =====================================================================
# Contract Plan — 50 contracts across 10 categories
# =====================================================================
# Shorthand for page guidance in prompts
SHORT = "Make it detailed enough to roughly equal 2-3 standard pages."
MEDIUM = "Make it detailed enough to roughly equal 4-6 standard pages."
LONG = "Make it extremely detailed, comparable to 8-12 pages."

contracts_plan = [
    # -----------------------------------------------------------------
    # NDAs (8 contracts)
    # -----------------------------------------------------------------
    {"type": "NDA", "filename": "NDA_Standard",
     "prompt": f"Generate a standard unilateral Non-Disclosure Agreement between a technology company and a potential investor. {SHORT}"},

    {"type": "NDA", "filename": "NDA_Mutual",
     "prompt": f"Generate a Mutual Non-Disclosure Agreement between two technology companies exploring a joint product development partnership. {SHORT}"},

    {"type": "NDA", "filename": "NDA_Employee_Onboarding",
     "prompt": f"Generate an Employee Non-Disclosure and Confidentiality Agreement for a new hire at a pharmaceutical company. Include IP assignment and non-solicitation clauses. {SHORT}"},

    {"type": "NDA", "filename": "NDA_Vendor_Onboarding",
     "prompt": f"Generate a Vendor Non-Disclosure Agreement for a healthcare IT vendor being onboarded to access patient data systems. Include HIPAA compliance language. {MEDIUM}"},

    {"type": "NDA", "filename": "NDA_Multilateral",
     "prompt": f"Generate a Multilateral Non-Disclosure Agreement among three companies forming a consortium for a government infrastructure bid. {MEDIUM}"},

    {"type": "NDA", "filename": "NDA_Departing_Employee",
     "prompt": f"Generate a Separation and Confidentiality Agreement for a departing senior executive. Include non-compete (2 years), non-solicitation, and return-of-materials clauses. {MEDIUM}"},

    {"type": "NDA", "filename": "NDA_Board_Advisor",
     "prompt": f"Generate a Board Advisor Confidentiality Agreement for a startup bringing on an external board advisor. Include equity vesting references. {SHORT}"},

    {"type": "NDA", "filename": "NDA_Merger_Due_Diligence",
     "prompt": f"Generate a Non-Disclosure Agreement for an M&A due diligence process between an acquiring company and a target company. Include data room access provisions and standstill clause. {MEDIUM}"},

    # -----------------------------------------------------------------
    # MSAs (7 contracts)
    # -----------------------------------------------------------------
    {"type": "MSA", "filename": "MSA_Complex",
     "prompt": f"Generate a highly complex Master Service Agreement for enterprise systems integration. Include dense SLA definitions, liability terms, escalation procedures, and change management. {LONG}"},

    {"type": "MSA", "filename": "MSA_With_Outlier",
     "prompt": f"Generate a complex Master Service Agreement for enterprise IT services. {LONG} CRITICAL INSTRUCTION: Inject an 'outlier' clause buried in the middle of the document. This outlier must be a massive $50M liability clause completely favoring the vendor without limits. Write it realistically to camouflage within standard boilerplate legal text."},

    {"type": "MSA", "filename": "MSA_IT_Staffing",
     "prompt": f"Generate a Master Service Agreement for IT staff augmentation services. Include rate cards, bench time policies, replacement guarantees, and background check requirements. {MEDIUM}"},

    {"type": "MSA", "filename": "MSA_Construction",
     "prompt": f"Generate a Master Service Agreement for commercial construction management services. Include safety standards, insurance requirements (min $2M general liability), bonding provisions, and change order procedures. {LONG}"},

    {"type": "MSA", "filename": "MSA_Cloud_Services",
     "prompt": f"Generate a Master Service Agreement for cloud infrastructure services (IaaS/PaaS). Include data sovereignty, uptime guarantees (99.95%), disaster recovery RPO/RTO, and data deletion on termination. {LONG}"},

    {"type": "MSA", "filename": "MSA_Consulting_Advisory",
     "prompt": f"Generate a Master Service Agreement for management consulting and advisory services. Include hourly/daily rate structures, travel expense policies, deliverable acceptance criteria, and intellectual property ownership. {MEDIUM}"},

    {"type": "MSA", "filename": "MSA_Logistics",
     "prompt": f"Generate a Master Service Agreement for third-party logistics (3PL) and warehousing services. Include storage rates, shipping SLAs, inventory accuracy guarantees (99.5%), and damage/loss liability. {LONG}"},

    # -----------------------------------------------------------------
    # Vendor & Supply Agreements (7 contracts)
    # -----------------------------------------------------------------
    {"type": "Vendor_Agreement", "filename": "Vendor_Agreement_IT_Hardware",
     "prompt": f"Generate a comprehensive Vendor Agreement for enterprise IT hardware procurement (servers, networking equipment). Include warranty terms, delivery schedules, acceptance testing, and volume discount tiers. {MEDIUM}"},

    {"type": "Vendor_Agreement", "filename": "Vendor_Agreement_Office_Supplies",
     "prompt": f"Generate a Vendor Agreement for office supplies and consumables with a national distributor. Include pricing catalogs, delivery SLAs, minimum order quantities, and annual price adjustment caps (max 3%). {SHORT}"},

    {"type": "Supply_Agreement", "filename": "Supply_Agreement_Raw_Materials",
     "prompt": f"Generate a Supply Agreement for industrial raw materials (steel, aluminum) between a manufacturer and a metals supplier. Include Incoterms (FOB), quality specifications, force majeure, and price indexing to commodity markets. {LONG}"},

    {"type": "Supply_Agreement", "filename": "Supply_Agreement_Medical_Devices",
     "prompt": f"Generate a Supply Agreement for FDA-regulated medical devices. Include regulatory compliance, quality audit rights, recall procedures, batch traceability, and product liability insurance ($5M minimum). {LONG}"},

    {"type": "Vendor_Agreement", "filename": "Vendor_Agreement_Catering",
     "prompt": f"Generate a Vendor Agreement for corporate catering and food services. Include menu pricing, health/safety compliance, staffing requirements, and cancellation policies. {SHORT}"},

    {"type": "Supply_Agreement", "filename": "Supply_Agreement_Electronics",
     "prompt": f"Generate a Supply Agreement for electronic components (semiconductors, PCBs) between an OEM and a distributor. Include lead time commitments, obsolescence management, and counterfeit part prevention. {MEDIUM}"},

    {"type": "Vendor_Agreement", "filename": "Vendor_Agreement_Security_Services",
     "prompt": f"Generate a Vendor Agreement for physical security guard services for a corporate campus. Include staffing levels, background check requirements, armed/unarmed specifications, and incident reporting procedures. {MEDIUM}"},

    # -----------------------------------------------------------------
    # Statements of Work (7 contracts)
    # -----------------------------------------------------------------
    {"type": "SOW", "filename": "SOW_Web_Development",
     "prompt": f"Generate a Statement of Work for a custom web application development project. Include detailed milestones, sprint schedule, acceptance criteria, UAT process, and fixed-price payment schedule totaling $285,000. {MEDIUM}"},

    {"type": "SOW", "filename": "SOW_Data_Migration",
     "prompt": f"Generate a Statement of Work for an enterprise data migration from on-premise Oracle to AWS cloud. Include data mapping, validation procedures, rollback plan, downtime windows, and a budget of $420,000. {MEDIUM}"},

    {"type": "SOW", "filename": "SOW_Cybersecurity_Audit",
     "prompt": f"Generate a Statement of Work for a comprehensive cybersecurity audit and penetration testing engagement. Include scope (network, application, social engineering), reporting deliverables, remediation support, and a fee of $95,000. {MEDIUM}"},

    {"type": "SOW", "filename": "SOW_Infrastructure_Upgrade",
     "prompt": f"Generate a Statement of Work for a data center infrastructure upgrade including server refresh, network modernization, and storage expansion. Budget: $1.2M. Include phased rollout and business continuity requirements. {LONG}"},

    {"type": "SOW", "filename": "SOW_Training_Program",
     "prompt": f"Generate a Statement of Work for developing and delivering a corporate training program on compliance and data privacy (GDPR/CCPA). Include curriculum development, instructor-led sessions, e-learning modules, and per-session pricing. {SHORT}"},

    {"type": "SOW", "filename": "SOW_ERP_Implementation",
     "prompt": f"Generate a Statement of Work for an SAP S/4HANA ERP implementation for a mid-size manufacturing company. Include discovery, configuration, data migration, integration, training, go-live support, and a total budget of $3.8M over 18 months. {LONG}"},

    {"type": "SOW", "filename": "SOW_Mobile_App",
     "prompt": f"Generate a Statement of Work for developing a cross-platform mobile application for field service management. Include wireframes phase, development sprints, QA testing, app store submission, and a budget of $175,000. {MEDIUM}"},

    # -----------------------------------------------------------------
    # SLAs (5 contracts)
    # -----------------------------------------------------------------
    {"type": "SLA", "filename": "SLA_Cloud_Hosting",
     "prompt": f"Generate a Service Level Agreement for cloud hosting services. Include uptime guarantees (99.99%), latency thresholds, incident response times (P1: 15min, P2: 1hr, P3: 4hr, P4: 24hr), service credits, and escalation matrix. {MEDIUM}"},

    {"type": "SLA", "filename": "SLA_Managed_IT",
     "prompt": f"Generate a Service Level Agreement for managed IT services (helpdesk, desktop support, network monitoring). Include ticket response/resolution times, CSAT targets, monthly reporting requirements, and penalty/bonus structure. {MEDIUM}"},

    {"type": "SLA", "filename": "SLA_Network_Maintenance",
     "prompt": f"Generate a Service Level Agreement for enterprise network maintenance and support. Include preventive maintenance schedules, MTTR targets, spare parts availability, and 24/7 NOC coverage requirements. {MEDIUM}"},

    {"type": "SLA", "filename": "SLA_Data_Center",
     "prompt": f"Generate a Service Level Agreement for colocation data center services. Include power availability (N+1), cooling guarantees, physical security, cross-connect provisioning times, and environmental monitoring. {MEDIUM}"},

    {"type": "SLA", "filename": "SLA_Print_Services",
     "prompt": f"Generate a Service Level Agreement for managed print services covering a 500-person office. Include device uptime, toner replenishment SLAs, per-page pricing, and quarterly usage reporting. {SHORT}"},

    # -----------------------------------------------------------------
    # Purchase Orders (4 contracts)
    # -----------------------------------------------------------------
    {"type": "Purchase_Order", "filename": "PO_Server_Hardware",
     "prompt": f"Generate a Purchase Order for enterprise server hardware (20 rack servers, networking switches, UPS units) totaling $487,500. Include itemized pricing, delivery terms, warranty details, and installation requirements. {SHORT}"},

    {"type": "Purchase_Order", "filename": "PO_Office_Furniture",
     "prompt": f"Generate a Purchase Order for office furniture for a new 150-person office build-out (desks, chairs, conference tables, storage). Total: $312,000. Include delivery schedule, assembly services, and damage policy. {SHORT}"},

    {"type": "Purchase_Order", "filename": "PO_Software_Licenses",
     "prompt": f"Generate a Purchase Order for enterprise software licenses (500 seats of productivity suite, 50 seats of project management tool, 20 seats of design software). Total: $245,000 annual. Include license types and renewal terms. {SHORT}"},

    {"type": "Purchase_Order", "filename": "PO_Fleet_Vehicles",
     "prompt": f"Generate a Purchase Order for a fleet of 25 commercial vehicles (vans and pickup trucks) for a field services company. Total: $875,000. Include vehicle specifications, delivery timeline, registration handling, and fleet warranty. {MEDIUM}"},

    # -----------------------------------------------------------------
    # Amendments & Change Orders (4 contracts)
    # -----------------------------------------------------------------
    {"type": "Amendment", "filename": "Amendment_MSA_Scope_Extension",
     "prompt": f"Generate an Amendment to an existing Master Service Agreement that extends the contract term by 2 years, adds a new service category (cybersecurity monitoring), and increases the liability cap from $5M to $8M. Reference the original MSA dated March 15, 2023. {SHORT}"},

    {"type": "Amendment", "filename": "Amendment_Price_Escalation",
     "prompt": f"Generate an Amendment to a Supply Agreement that implements a 7.5% price escalation due to raw material cost increases, effective retroactively. Include cost justification language and a price review mechanism every 6 months. {SHORT}"},

    {"type": "Change_Order", "filename": "Change_Order_Construction",
     "prompt": f"Generate a Construction Change Order that adds a new building wing (15,000 sq ft) to an existing construction project. Include revised timeline (+4 months), additional cost ($2.1M), updated milestones, and impact on existing warranties. {MEDIUM}"},

    {"type": "Amendment", "filename": "Amendment_Auto_Renewal_Hidden",
     "prompt": f"Generate an Amendment to an existing Vendor Agreement that adds several minor administrative updates (notice address change, updated contact persons, invoice format change). CRITICAL INSTRUCTION: Bury within these mundane changes a clause that converts the contract from a fixed 2-year term to auto-renewing annual terms with only a 15-day opt-out window. Make this clause look routine and unremarkable. {SHORT}"},

    # -----------------------------------------------------------------
    # Licensing Agreements (4 contracts)
    # -----------------------------------------------------------------
    {"type": "License", "filename": "License_Enterprise_Software",
     "prompt": f"Generate an Enterprise Software License Agreement for a CRM platform. Include perpetual vs subscription options, seat-based pricing, audit rights, source code escrow, and maintenance/support terms. {LONG}"},

    {"type": "License", "filename": "License_SaaS_Subscription",
     "prompt": f"Generate a SaaS Subscription Agreement for a cloud-based HR management platform. Include per-user monthly pricing, data ownership, API access terms, uptime SLA, and data export on termination. {MEDIUM}"},

    {"type": "License", "filename": "License_IP_Technology",
     "prompt": f"Generate a Technology License Agreement granting rights to use patented manufacturing process technology. Include royalty structure (5% of net revenue), field-of-use restrictions, sublicensing rights, and improvement clauses. {MEDIUM}"},

    {"type": "License", "filename": "License_IP_Broad_Assignment",
     "prompt": f"Generate a Technology License and IP Assignment Agreement for a software development engagement. CRITICAL INSTRUCTION: Include a clause buried in the IP section that assigns ALL intellectual property created by the client's employees while using the licensed software to the licensor, not just derivatives. Make it read like standard work-product language so it does not stand out. {MEDIUM}"},

    # -----------------------------------------------------------------
    # Consulting & Professional Services (4 contracts)
    # -----------------------------------------------------------------
    {"type": "Consulting", "filename": "Consulting_Strategy",
     "prompt": f"Generate a Strategic Consulting Engagement Agreement for a management consulting firm advising on digital transformation. Include phased approach, executive sponsor requirements, deliverable schedule, and fees ($450/hr partner, $275/hr senior consultant). {MEDIUM}"},

    {"type": "Consulting", "filename": "Consulting_Financial_Advisory",
     "prompt": f"Generate a Financial Advisory Engagement Letter for M&A advisory services. Include success fee structure (2% of transaction value), retainer ($50K/month), tail period (18 months), and exclusivity provisions. {MEDIUM}"},

    {"type": "Consulting", "filename": "Independent_Contractor_Developer",
     "prompt": f"Generate an Independent Contractor Agreement for a senior software developer. Include hourly rate ($165/hr), IP assignment, non-compete (limited to 6 months), equipment provisions, and IRS classification safeguards. {SHORT}"},

    {"type": "Consulting", "filename": "Professional_Services_Legal_Review",
     "prompt": f"Generate a Professional Services Agreement for an outside law firm providing contract review and compliance advisory services. Include blended hourly rates, alternative fee arrangements for volume work, conflict of interest provisions, and privilege protections. {MEDIUM}"},

    # -----------------------------------------------------------------
    # Specialty Contracts (5 contracts)
    # -----------------------------------------------------------------
    {"type": "Lease", "filename": "Lease_Heavy_Equipment",
     "prompt": f"Generate an Equipment Lease Agreement for heavy construction machinery (excavators, cranes, loaders). Include monthly lease rates, maintenance responsibilities, insurance requirements, return conditions, and purchase option at end of term. {MEDIUM}"},

    {"type": "Subcontractor", "filename": "Subcontractor_Electrical",
     "prompt": f"Generate a Subcontractor Agreement for electrical installation work on a commercial building project. Include scope flow-down from prime contract, pay-when-paid terms, safety compliance, lien waiver requirements, and retainage (10%). {MEDIUM}"},

    {"type": "Teaming", "filename": "Teaming_Agreement_Government",
     "prompt": f"Generate a Teaming Agreement between two companies bidding on a federal government IT modernization contract (FAR/DFARS compliant). Include workshare split (60/40), small business set-aside provisions, organizational conflict of interest, and exclusivity during bid process. {MEDIUM}"},

    {"type": "Warranty", "filename": "Warranty_Manufacturing",
     "prompt": f"Generate a Product Warranty Agreement for industrial manufacturing equipment. Include standard warranty (2 years parts and labor), extended warranty options, exclusions, RMA procedures, and performance guarantees (minimum 95% uptime). {MEDIUM}"},

    {"type": "Joint_Venture", "filename": "Joint_Venture_Real_Estate",
     "prompt": f"Generate a Joint Venture Agreement for a commercial real estate development project (mixed-use building). Include capital contribution schedule ($15M total, 60/40 split), profit/loss distribution, management committee structure, deadlock resolution, and exit/buyout provisions. CRITICAL INSTRUCTION: Include a one-sided indemnification clause that requires only one partner to indemnify the other for all project losses regardless of fault, but write it in dense legal language so it appears balanced at first glance. {LONG}"},
]

# -----------------------------
# Processing Pipeline
# -----------------------------
def process_contract(contract):
    text = generate_contract(contract["prompt"])

    pdf_path = os.path.join(PDF_DIR, contract["filename"] + ".pdf")
    save_pdf(text, pdf_path)

    chunks = chunk_text(text)
    metadata = {
        "document_name": contract["filename"],
        "type": contract["type"],
        "chunk_count": len(chunks),
        "chunks": [
            {"chunk_id": f"{contract['filename']}_{i}", "text": chunk}
            for i, chunk in enumerate(chunks)
        ],
    }

    meta_path = os.path.join(META_DIR, contract["filename"] + ".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return contract["filename"]


# -----------------------------
# Main
# -----------------------------
def main():
    # Skip contracts that already have a PDF (so you can re-run after failures)
    remaining = [
        c for c in contracts_plan
        if not os.path.exists(os.path.join(PDF_DIR, c["filename"] + ".pdf"))
    ]

    print(f"\n{len(contracts_plan)} total contracts, {len(remaining)} remaining to generate.\n")

    if not remaining:
        print("All contracts already generated. Delete PDFs to regenerate.")
        return

    failed = []
    for i, contract in enumerate(tqdm(remaining, unit="contract")):
        try:
            process_contract(contract)
            tqdm.write(f"  Done: {contract['filename']}")
        except Exception as e:
            failed.append(contract["filename"])
            tqdm.write(f"  FAILED: {contract['filename']} — {e}")

        # Rate limit pause (skip after last one)
        if i < len(remaining) - 1:
            time.sleep(DELAY_BETWEEN_CONTRACTS)

    print(f"\nGeneration complete. {len(remaining) - len(failed)} succeeded, {len(failed)} failed.")
    if failed:
        print(f"Failed contracts: {failed}")
        print("Re-run the script to retry only the failed ones.")


if __name__ == "__main__":
    main()
