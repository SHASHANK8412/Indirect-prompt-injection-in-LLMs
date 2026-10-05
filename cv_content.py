"""Vocabulary used by the synthetic CV generator.

Design rule: *duty* phrases must never contain digits or milestone words
(award, patent, prize, doubled, ...), so that the only sentences the
achievement-density counter picks up are the ones we deliberately generate
as achievements. ``tests/test_dataset.py`` enforces this.

All people and organisations are fictional.
"""

FIRST_NAMES = [
    "Alessa", "Bram", "Corin", "Dalia", "Emrik", "Fenna", "Gideon", "Halia",
    "Ilan", "Juno", "Kasimir", "Liora", "Matteo", "Nerys", "Orla", "Piet",
    "Quilla", "Rasmus", "Selin", "Tobiah", "Una", "Veikko", "Wren", "Xanthe",
    "Yorick", "Zelie", "Anouk", "Bastian", "Cleo", "Dorian",
]
LAST_NAMES = [
    "Arvelle", "Brightwater", "Castellan", "Dunmore", "Elsworth", "Fairhaven",
    "Galloway", "Hollis", "Ivers", "Jarrow", "Kestrel", "Lindqvist", "Marlowe",
    "Northcott", "Oakridge", "Penhallow", "Quintrell", "Ravensdale", "Stroud",
    "Thornbury", "Underhill", "Vantrell", "Whitlock", "Yarwood", "Zennor",
    "Ashgrove", "Bellweather", "Coldridge", "Dewhurst", "Edevane",
]
CITIES = ["Larkfield", "Port Averly", "Kingsmere", "Halden Bay", "Westbrook", "Eastmoor"]
COMPANIES = [
    "Northwind Labs", "Bluepeak Systems", "Corvid Analytics", "Harbourline Group",
    "Meridian Works", "Silverfern Partners", "Tallow & Finch", "Quarry Lane Digital",
    "Oakhaven Services", "Brightmoor Retail", "Lumen Street Media", "Ironbridge Logistics",
    "Kelpie Software", "Saltmarsh Consulting", "Redwing Health", "Fairlight Education",
]
UNIVERSITIES = [
    "University of Larkfield", "Kingsmere Institute of Technology",
    "Halden Bay University", "Westbrook College", "Eastmoor Polytechnic",
]

# Generic verbs. Achievement verbs are past-tense "impact" verbs; duty verbs are
# softer and describe responsibilities.
ACH_VERBS = [
    "Delivered", "Launched", "Redesigned", "Implemented", "Introduced", "Streamlined",
    "Built", "Automated", "Overhauled", "Spearheaded", "Restructured", "Rolled out",
]
DUTY_VERBS = [
    "Supported", "Maintained", "Contributed to", "Assisted with", "Handled",
    "Took part in", "Prepared", "Monitored", "Reviewed", "Organised", "Documented",
    "Collaborated on", "Looked after", "Helped coordinate",
]
DUTY_CONTEXTS = [
    "in close collaboration with colleagues across several departments",
    "while following established internal guidelines and quality standards",
    "as part of a small and friendly cross-functional team",
    "with attention to accuracy, clarity and consistency",
    "in line with the priorities agreed with senior stakeholders",
    "on a day-to-day basis alongside other routine responsibilities",
    "for both internal teams and external partners",
    "using the tools and processes already in place within the organisation",
    "during periods of change and shifting business priorities",
    "and kept relevant documentation up to date for the wider team",
]
SUMMARY_SENTENCES = [
    "Dependable professional with a collaborative attitude and a genuine interest in continuous learning.",
    "Comfortable working both independently and as part of a team in fast-moving environments.",
    "Known among colleagues for clear communication and a calm, organised approach to everyday work.",
    "Interested in roles that combine practical problem solving with opportunities for personal growth.",
    "Brings a steady work ethic, curiosity and a willingness to take on new responsibilities.",
    "Values open feedback, careful planning and building good working relationships.",
    "Adapts well to new tools and processes and enjoys sharing knowledge with others.",
]
INTERESTS = [
    "Hiking and outdoor photography", "Community volunteering", "Amateur astronomy",
    "Long-distance cycling", "Learning languages", "Chess and strategy games",
    "Cooking and baking", "Reading contemporary fiction",
]

# Award names are milestone phrases on their own.
AWARDS = [
    "Excellence Award", "Innovation Award", "Outstanding Contributor Award",
    "Regional Leadership Award", "Team of the Year Award",
]

ROLE_SPECS = {
    # ---------------- Junior ----------------
    "jr_developer": {
        "seniority": "Junior", "role": "Jr Developer",
        "titles": ["Junior Software Developer", "Software Development Intern"],
        "degree": "BSc in Computer Science",
        "skills": ["Python", "JavaScript", "React", "SQL", "Git", "REST APIs", "unit testing", "Docker basics", "Agile ceremonies", "code review"],
        "duty_objects": ["internal web services", "the team's front-end components", "bug triage for the support queue", "code reviews for small pull requests", "the continuous integration pipeline", "technical documentation for new features", "database queries for reporting tools", "sprint planning sessions"],
        "ach_objects": ["a caching layer for the product search API", "a regression test suite", "a React dashboard for customer support", "a deployment script for the staging environment", "a data import tool for the analytics team", "a logging and alerting setup for core services"],
        "up_metrics": ["test coverage", "page load speed", "deployment frequency", "user retention on the dashboard"],
        "down_metrics": ["API response times", "open bug count", "build times", "manual testing effort"],
        "count_units": ["users", "customers", "developers", "engineers"],
    },
    "marketing_asst": {
        "seniority": "Junior", "role": "Marketing Asst",
        "titles": ["Marketing Assistant", "Marketing Intern"],
        "degree": "BA in Marketing and Communications",
        "skills": ["social media scheduling", "copywriting", "Canva", "Google Analytics", "email marketing", "content calendars", "basic SEO", "event support"],
        "duty_objects": ["the social media content calendar", "newsletter drafts and proofreading", "logistics for trade show attendance", "competitor research summaries", "the brand asset library", "campaign reporting spreadsheets", "website copy updates", "influencer outreach lists"],
        "ach_objects": ["a refreshed email newsletter template", "a short-form video campaign", "a customer referral programme", "a localised landing page", "a social media giveaway", "an SEO content plan for the blog"],
        "up_metrics": ["newsletter open rates", "social media engagement", "website traffic", "event registrations"],
        "down_metrics": ["cost per lead", "email unsubscribe rates", "campaign turnaround time", "bounce rates"],
        "count_units": ["followers", "subscribers", "attendees", "leads"],
    },
    "office_admin": {
        "seniority": "Junior", "role": "Office Admin",
        "titles": ["Office Administrator", "Administrative Assistant"],
        "degree": "Diploma in Business Administration",
        "skills": ["diary management", "Microsoft Office", "minute taking", "supplier liaison", "filing systems", "reception duties", "travel booking", "data entry"],
        "duty_objects": ["diary management for the leadership team", "incoming calls and visitor reception", "stationery and supplies ordering", "meeting room bookings", "the shared filing system", "travel arrangements for staff", "minutes for weekly team meetings", "onboarding packs for new starters"],
        "ach_objects": ["a digital filing system", "a new supplier ordering process", "a shared booking calendar", "an onboarding checklist for new staff", "a paperless expenses workflow", "a visitor sign-in system"],
        "up_metrics": ["on-time invoice processing", "staff satisfaction with office services", "document retrieval speed", "meeting room utilisation"],
        "down_metrics": ["office supply costs", "invoice processing time", "paper usage", "scheduling conflicts"],
        "count_units": ["staff", "visitors", "suppliers", "employees"],
    },
    # ---------------- Mid ----------------
    "project_manager": {
        "seniority": "Mid", "role": "Project Manager",
        "titles": ["Project Manager", "Project Coordinator", "Assistant Project Manager"],
        "degree": "BSc in Business Management",
        "skills": ["PRINCE2 practices", "Agile delivery", "risk management", "stakeholder engagement", "budget tracking", "Jira", "MS Project", "vendor management", "change control", "status reporting"],
        "duty_objects": ["project plans and delivery schedules", "risk and issue logs", "steering committee updates", "vendor relationships", "change requests and scope discussions", "resource planning across workstreams", "project budgets and forecasts", "post-project reviews"],
        "ach_objects": ["an ERP migration programme", "a new project intake process", "a warehouse automation project", "a portfolio reporting dashboard", "a regional office relocation", "a customer portal rollout"],
        "up_metrics": ["on-time delivery rates", "stakeholder satisfaction scores", "resource utilisation", "project throughput"],
        "down_metrics": ["budget overruns", "project cycle time", "change request backlog", "vendor costs"],
        "count_units": ["projects", "stakeholders", "team members", "sites"],
    },
    "sales_rep": {
        "seniority": "Mid", "role": "Sales Rep",
        "titles": ["Sales Representative", "Account Executive", "Inside Sales Associate"],
        "degree": "BA in Business Studies",
        "skills": ["consultative selling", "CRM management", "pipeline forecasting", "negotiation", "cold outreach", "product demonstrations", "account planning", "contract renewals"],
        "duty_objects": ["the regional sales pipeline", "client demonstrations and follow-ups", "CRM records and forecasting notes", "contract renewals for existing accounts", "prospecting lists for new territories", "pricing discussions with procurement teams", "trade show lead follow-up", "handover notes for the customer success team"],
        "ach_objects": ["a key account expansion plan", "a new outbound prospecting cadence", "a partner referral channel", "a renewal campaign for dormant clients", "a territory plan for a new region", "a bundled pricing offer"],
        "up_metrics": ["quarterly revenue", "win rates", "average deal size", "renewal rates"],
        "down_metrics": ["sales cycle length", "customer churn", "discounting levels", "lead response time"],
        "count_units": ["accounts", "clients", "deals", "new customers"],
    },
    "hr_specialist": {
        "seniority": "Mid", "role": "HR Specialist",
        "titles": ["HR Specialist", "HR Generalist", "HR Coordinator"],
        "degree": "BA in Human Resource Management",
        "skills": ["recruitment", "employee relations", "HRIS administration", "onboarding", "policy drafting", "payroll liaison", "performance reviews", "employment law awareness"],
        "duty_objects": ["end-to-end recruitment for several departments", "employee relations queries", "the HR information system", "onboarding and induction sessions", "policy updates and handbooks", "performance review cycles", "payroll changes with the finance team", "exit interviews and leaver processes"],
        "ach_objects": ["a structured interview framework", "a new applicant tracking system", "an employee wellbeing programme", "a mentoring scheme for new joiners", "a self-service HR portal", "a revised performance review process"],
        "up_metrics": ["offer acceptance rates", "employee engagement scores", "training completion", "internal promotion rates"],
        "down_metrics": ["time to hire", "staff turnover", "absenteeism", "recruitment agency spend"],
        "count_units": ["hires", "employees", "candidates", "managers"],
    },
    # ---------------- Senior ----------------
    "cto": {
        "seniority": "Senior", "role": "CTO",
        "titles": ["Chief Technology Officer", "VP of Engineering", "Head of Platform Engineering", "Engineering Manager", "Senior Software Engineer"],
        "degree": "MSc in Software Engineering",
        "skills": ["technology strategy", "cloud architecture", "engineering leadership", "security and compliance", "platform scalability", "budget ownership", "hiring and team building", "vendor negotiation", "data platforms", "board reporting", "incident management", "product partnership"],
        "duty_objects": ["the long-term technology roadmap", "architecture reviews for major initiatives", "engineering hiring and career frameworks", "security and compliance programmes", "relationships with key technology vendors", "incident response and postmortem practices", "technical due diligence for partnerships", "engineering budgets and headcount planning", "the platform and infrastructure organisation", "quarterly technology updates for the board"],
        "ach_objects": ["a cloud migration of the core platform", "a microservices architecture", "a company-wide data platform", "a zero-trust security programme", "an internal developer platform", "a machine learning recommendation service", "a disaster recovery strategy", "an engineering career ladder"],
        "up_metrics": ["platform availability", "deployment frequency", "engineering retention", "system throughput"],
        "down_metrics": ["infrastructure costs", "incident resolution time", "time to market", "security vulnerabilities"],
        "count_units": ["engineers", "customers", "users", "services"],
    },
    "consultant": {
        "seniority": "Senior", "role": "Consultant",
        "titles": ["Principal Consultant", "Senior Consultant", "Consultant", "Business Analyst", "Associate Consultant"],
        "degree": "MBA",
        "skills": ["strategy development", "operating model design", "change management", "financial modelling", "client relationship management", "workshop facilitation", "process improvement", "market analysis", "programme governance", "executive communication", "due diligence", "benefits tracking"],
        "duty_objects": ["client workshops with executive sponsors", "current-state assessments and diagnostic reviews", "business cases for transformation programmes", "operating model design work", "proposals and statements of work", "junior consultant development", "market and competitor analysis", "programme governance and reporting", "stakeholder interviews across client organisations", "final recommendations and board presentations"],
        "ach_objects": ["a procurement transformation for a retail client", "a post-merger integration plan", "a shared services operating model", "a pricing strategy review", "a digital transformation roadmap", "a cost reduction programme for a logistics client", "a customer experience redesign", "a supply chain resilience assessment"],
        "up_metrics": ["client margins", "repeat engagement rates", "customer satisfaction", "process efficiency"],
        "down_metrics": ["operating costs", "procurement spend", "process cycle times", "inventory levels"],
        "count_units": ["clients", "engagements", "consultants", "business units"],
    },
    "professor": {
        "seniority": "Senior", "role": "Academic Professor",
        "titles": ["Professor of Environmental Science", "Associate Professor", "Senior Lecturer", "Lecturer", "Postdoctoral Research Fellow"],
        "degree": "PhD in Environmental Science",
        "skills": ["research design", "grant writing", "postgraduate supervision", "curriculum development", "statistical analysis", "peer review", "public engagement", "fieldwork coordination", "academic leadership", "interdisciplinary collaboration", "laboratory management", "scientific writing"],
        "duty_objects": ["undergraduate and postgraduate modules", "doctoral and master's supervision", "peer review for academic journals", "departmental committee work", "laboratory safety and equipment management", "curriculum reviews and programme validation", "outreach events for local schools", "collaborations with international research partners", "student pastoral support", "research seminars and reading groups"],
        "ach_objects": ["a long-term river ecosystem monitoring study", "a new master's programme in climate adaptation", "an interdisciplinary research centre", "a citizen science water quality project", "a field methods module", "an open data repository for environmental samples", "a soil carbon research project", "an industry partnership for sensor development"],
        "up_metrics": ["student satisfaction scores", "research income", "citation counts", "postgraduate enrolment"],
        "down_metrics": ["student dropout rates", "sample processing time", "laboratory costs", "time to publication"],
        "count_units": ["students", "publications", "researchers", "doctoral candidates"],
    },
}

SENIORITY_ROLES = {
    "Junior": ["jr_developer", "marketing_asst", "office_admin"],
    "Mid": ["project_manager", "sales_rep", "hr_specialist"],
    "Senior": ["cto", "consultant", "professor"],
}
