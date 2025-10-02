GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly", 
    "https://www.googleapis.com/auth/gmail.modify"   
]

GMAIL_CLIENT_SECRET_FILE = "config/client_secret.json"
GMAIL_TOKEN_FILE = "config/token.json"              


GMAIL_EMAIL_QUERY = 'has:attachment (filename:pdf OR filename:doc OR filename:docx) (cv OR resume)'


GMAIL_MAX_FETCH = 500 


CHECK_INTERVAL_SECONDS = 300 


SHEETS_ID = "1zJN3243aDq5SYN64H69GgeQkwwsZbBip7So-hwLxHv8"# mine "1D84molgG1bGT1EX4B9BLu4EgrdrFh3TFvyxgi1X2l10" #"1BsCSX_JfuXNrVh1YfrGSq2X9rGvedOL3tcBYFfZYgSY" 

SHEETS_TAB_NAME = "Test_Application" #"Test_Application_1" #"Applicants"

SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


SHEETS_SERVICE_ACCOUNT_FILE = "config/credentials.json" 


DRIVE_FOLDER_ID = "0AJ7tifipg8B1Uk9PVA"#"1-itV98LZInDZZWwLjkSrcewcZlmOLoRw"#"1DFM6XOBG8w26_91ttdAbOG9DyXTTtdBd"


#DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"] 
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]


ALLOWED_RESUME_EXTENSIONS = {".pdf", ".doc", ".docx"}


AUTO_REPLY_SUBJECT = "Application Received - LinkPlus IT"

AUTO_REPLY_BODY_PLAIN = """
Dear {applicant_name},

Thank you for your interest in our company and for submitting your resume.
We have successfully received your application and it is currently being reviewed.

We appreciate you taking the time to apply. If your qualifications match our requirements,
we will be in touch regarding the next steps in the hiring process.

Due to the high volume of applications, we are unable to respond to each candidate individually
unless they are selected for an interview. We appreciate your understanding.

Sincerely,

The LinkPlus IT Hiring Team
"""

DEFAULT_JOB_TITLE = "General Application"

DEFAULT_SEMANTIC_THRESHOLD = 0.70


MIN_TOKENS_FOR_SEMANTIC = 3


JOB_KEYWORDS_MAPPING = {

    "Java Developer": ["java developer", "java engineer", "spring developer", "spring boot developer", "spring boot", "java backend", "j2ee developer", "enterprise java", "hibernate", "maven", "gradle", "java junior", "java software engineer"],
    ".NET Developer": ["dotnet developer", ".net developer", ".net core developer", "c# developer", "c# engineer", "asp.net developer", "windows forms developer", "wpf developer", "xamarin developer"],
    "Python Developer": ["python developer", "python engineer", "django developer", "flask developer", "fastapi developer", "python backend", "python web"],
    "Node.js Developer": ["node developer", "node.js developer", "nodejs developer", "express.js", "nest.js", "nest.js developer", "javascript backend"],
    "Go Developer": ["go developer", "golang developer", "go engineer", "golang engineer"],
    "Ruby on Rails Developer": ["ruby on rails developer", "rails developer", "ruby developer"],
    "PHP Developer": ["php developer", "laravel developer", "symfony developer", "wordpress developer"],
    "Rust Developer": ["rust developer", "rust engineer"],
    "C++ Developer": ["c++ developer", "cpp developer", "c++ engineer", "game developer", "embedded c++"],
    "C Developer": ["c developer", "embedded c"],
    "Kotlin Developer": ["kotlin developer", "kotlin backend"],
    "Scala Developer": ["scala developer", "akka", "spark scala"],
    "Perl Developer": ["perl developer"],

    "Mobile Developer (iOS)": ["ios developer", "swift developer", "objective-c developer", "xcode", "apple developer", "ios engineer", "apple engineer"], # Added "ios engineer", "apple engineer"
    "Mobile Developer (Android)": ["android developer", "kotlin android", "java android", "android studio", "google android developer", "android engineer", "google engineer"], # Added "android engineer", "google engineer"
    "Mobile Developer (Cross-Platform)": ["mobile developer", "cross-platform developer", "react native developer", "flutter developer", "ionic developer", "xamarin developer", "hybrid mobile"],

    "Backend Developer": ["backend developer", "back-end developer", "backend engineer", "back-end engineer", "server-side developer", "api developer", "microservices developer", "distributed systems developer", "web backend"],
    "Frontend Developer": ["frontend developer", "front-end developer", "frontend engineer", "front-end engineer", "web developer", "web engineer", "ui developer", "ui engineer", "ux engineer", "user interface developer", "client-side developer", "js developer", "javascript developer", "typescript developer"],
    "Fullstack Developer": ["fullstack developer", "full-stack developer", "fullstack engineer", "full-stack engineer", "end-to-end developer", "mern stack", "mean stack", "mevn stack", "lamp stack"],


    "Junior Software Engineer": ["junior software engineer", "junior software developer", "entry-level developer", "entry-level engineer", "associate software engineer", "associate developer", "grad developer", "graduate engineer"],
    "Software Engineer": ["software engineer", "software developer", "developer", "dev", "programmer", "engineer", "coding", "programming", "applications developer", "application engineer", "computer engineer"],
    "Senior Software Engineer": ["senior software engineer", "senior software developer", "lead software engineer", "principal software engineer", "staff software engineer", "architect software", "tech lead", "technical lead", "lead developer", "principal developer", "staff developer", "senior dev"],
    "Software Architect": ["software architect", "solution architect", "enterprise architect", "technical architect", "system architect", "cloud architect"],


    "Data Scientist": ["data scientist", "machine learning engineer", "ml engineer", "deep learning engineer", "ai engineer", "artificial intelligence engineer", "data modeler", "statistical modeler", "predictive modeling", "nlp engineer", "computer vision engineer", "tensorflow", "pytorch", "scikit-learn", "keras", "rstudio", "pandas", "numpy", "jupyter", "spark"],
    "Data Analyst": ["data analyst", "business intelligence analyst", "bi analyst", "sql analyst", "data visualization specialist", "tableau developer", "power bi developer", "excel analyst", "data interpretation", "dashboarding", "reporting analyst", "statistics analyst"],
    "Data Engineer": ["data engineer", "etl developer", "big data engineer", "data pipeline", "data warehousing", "data lake", "spark", "hadoop", "kafka", "dbt", "airflow", "data bricks"],
    "MLOps Engineer": ["mlops engineer", "ml ops", "machine learning operations", "model deployment", "model monitoring", "ml pipeline"],
    "BI Developer": ["bi developer", "business intelligence developer", "data warehousing developer", "etl developer"],


    "DevOps Engineer": ["devops engineer", "devops", "site reliability engineer", "sre", "cloud engineer", "cloud infrastructure", "infrastructure engineer", "automation engineer", "ci/cd engineer"],
    "Cloud Engineer (AWS)": ["aws engineer", "aws solutions architect", "aws developer", "amazon web services"],
    "Cloud Engineer (Azure)": ["azure engineer", "azure developer", "microsoft azure"],
    "Cloud Engineer (GCP)": ["gcp engineer", "google cloud platform engineer", "google cloud developer"],
    "Kubernetes Specialist": ["kubernetes", "k8s", "container orchestration"],
    "Docker Specialist": ["docker", "containerization"],
    "System Administrator": ["system administrator", "sysadmin", "linux administrator", "windows administrator", "server administrator", "network administrator"],
    "Network Engineer": ["network engineer", "network specialist", "ccna", "ccnp", "routing", "switching", "firewall"],
    "Cybersecurity Engineer": ["cybersecurity engineer", "security engineer", "infosec engineer", "penetration tester", "vulnerability analyst", "soc analyst", "incident response", "security operations"],
    "Cloud Security Engineer": ["cloud security", "aws security", "azure security", "gcp security"],

    "QA Engineer": ["qa engineer", "quality assurance engineer", "software tester", "test automation engineer", "sdra", "sdwt", "manual tester", "automation tester"],
    "Test Automation Engineer": ["test automation engineer", "automation qa", "selenium automation", "cypress automation", "playwright automation", "api automation", "performance testing"],

    "Project Manager (Software)": ["project manager", "software project manager", "technical project manager", "agile project manager", "scrum master", "pmp", "prince2"],
    "Product Manager (Tech)": ["product manager", "product owner", "technical product manager", "agile product owner", "product strategy", "roadmap", "user stories", "market analysis"],
    "Engineering Manager": ["engineering manager", "software engineering manager", "development manager", "team lead manager"],


    "Blockchain Developer": ["blockchain developer", "solidity developer", "web3 developer", "ethereum developer", "smart contract developer"],
    "Embedded Systems Engineer": ["embedded systems engineer", "firmware engineer", "microcontroller", "rtos", "hardware-software"],
    "Game Developer": ["game developer", "unity developer", "unreal engine developer", "gamedev", "c# unity", "c++ game"],
    "ERP/CRM Consultant": ["erp consultant", "crm consultant", "sap consultant", "microsoft dynamics", "salesforce developer", "salesforce consultant"],
    "Technical Writer": ["technical writer", "documentation specialist", "api documentation"],
    "Salesforce Developer": ["salesforce developer", "apex developer", "lightning web components", "lwc developer"],


    "IT Support Specialist": ["it support", "help desk support", "technical support specialist", "desktop support"],
    "System Analyst": ["system analyst", "business system analyst"],
    "Database Administrator": ["database administrator", "dba", "sql dba", "nosql dba", "mongodb dba"],
    "UI/UX Designer": ["ui/ux designer", "ui designer", "ux designer", "user interface designer", "user experience designer", "product designer", "interaction designer", "visual designer", "web designer", "figma", "sketch", "adobe xd", "wireframing", "prototyping", "usability testing", "design system"],


    "Human Resources": ["hr", "human resources", "recruiter", "talent acquisition", "people operations"],
    "Marketing Specialist": ["marketing specialist", "digital marketing", "seo specialist", "content marketing", "social media manager"],
    "Sales Representative": ["sales representative", "account executive", "business development representative"],
    "General Application": ["general application", "resume", "cv", "job application", "career inquiry", "open position", "unspecified role", "not specified", "job", "career"],
}