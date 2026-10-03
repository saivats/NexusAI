import re
import string

SYNONYM_MAP = {
    "wifi": "wifi wireless internet network connection",
    "wi-fi": "wifi wireless internet network connection",
    "internet": "internet wifi network connection online",
    "network": "network internet wifi connection",
    "connection": "connection internet wifi network",
    "not working": "not working broken down issue problem",
    "slow": "slow lag laggy sluggish",
    "cant connect": "cant connect no connection unable connect",
    "can't connect": "cant connect no connection unable connect",
    "login failed": "login failed wrong password locked authentication",
    "locked out": "locked out account locked cant login",
    "locked": "locked account locked cant login locked out",
    "password": "password credentials login authentication reset",
    "vpn": "vpn virtual private network remote access",
    "email": "email mail outlook inbox",
    "laptop": "laptop computer device pc",
    "pc": "pc computer laptop device",
    "printer": "printer print printing paper",
    "leave": "leave time off vacation holiday absence chutti",
    "vacation": "vacation leave holiday time off pto",
    "holiday": "holiday vacation leave time off chutti",
    "salary": "salary pay wages payslip paycheck compensation",
    "payslip": "payslip salary pay wages paycheck",
    "paycheck": "paycheck salary pay wages payslip",
    "tuition": "tuition fee fees payment semester",
    "fee": "fee tuition payment charges cost fees",
    "fees": "fees tuition payment charges cost fee",
    "refund": "refund money back reimbursement return",
    "scholarship": "scholarship financial aid grant bursary funding",
    "hostel": "hostel dorm dormitory housing accommodation",
    "dorm": "dorm hostel dormitory housing accommodation",
    "room": "room accommodation space",
    "gym": "gym recreation fitness exercise workout sports",
    "library": "library study reading room books",
    "parking": "parking car vehicle permit lot",
    "shuttle": "shuttle bus transport transit ride",
    "exam": "exam examination test finals midterm assessment",
    "exams": "exams examination test finals midterm assessment",
    "grade": "grade grades gpa marks result score",
    "grades": "grades grade gpa marks results scores",
    "gpa": "gpa grade grades marks cgpa result",
    "marks": "marks grades grade gpa result score",
    "result": "result grade grades marks score",
    "results": "results grades marks scores gpa",
    "transcript": "transcript academic record marksheet grades history",
    "attendance": "attendance absent absence proxy bunk",
    "assignment": "assignment homework submission coursework project",
    "course": "course subject class module",
    "courses": "courses subjects classes modules registration",
    "register": "register enroll sign up registration",
    "admission": "admission admissions apply entry enrollment intake",
    "admissions": "admissions admission apply entry enrollment intake",
    "apply": "apply application register enroll",
    "visa": "visa immigration international student study permit",
    "orientation": "orientation induction welcome freshers onboarding",
    "placement": "placement internship career job industry",
    "internship": "internship placement career job work experience",
    "backlog": "backlog arrear supplementary reappear failed",
    "arrear": "arrear backlog supplementary reappear failed",
    "major": "major branch specialization stream department",
    "club": "club organization society extracurricular activities",
    "clubs": "clubs organizations societies extracurricular activities",
    "insurance": "insurance health medical coverage benefits",
    "broken": "broken fix repair damaged issue problem",
    "food": "food dining cafeteria canteen meal eat",
    "bus": "bus shuttle transport transit ride",
    "sick": "sick leave medical illness health",
    "overtime": "overtime extra hours work late",
    "resign": "resign resignation quit leaving job",
    "transfer": "transfer lateral entry switch college",
    "hall ticket": "hall ticket admit card exam entry",
    "admit card": "admit card hall ticket exam entry",
    "kab": "kab when date time schedule",
    "kaise": "kaise how process procedure steps",
    "kitna": "kitna how much amount total",
    "kaha": "kaha where location place",
    "kya": "kya what which",
    "bharni": "bharni pay submit fill",
    "milega": "milega get receive available when",
    "milegi": "milegi get receive available when",
    "chahiye": "chahiye need want require",
    "karna": "karna do perform action",
    "hoga": "hoga will happen when",
    "nahi": "nahi not no issue problem",
    "chutti": "chutti leave holiday vacation time off",
    "paisa": "paisa money payment fees cost",
    "paise": "paise money payment fees cost",
    "padhai": "padhai study course academics education",
    "parhai": "parhai study course academics education",
    "pariksha": "pariksha exam examination test",
    "pareeksha": "pareeksha exam examination test",
    "naukri": "naukri job placement career employment",
    "nokri": "nokri job placement career employment",
    "dakhila": "dakhila admission entry enrollment",
    "daakhila": "daakhila admission entry enrollment",
    "bimaari": "bimaari sick illness medical health",
    "beemar": "beemar sick illness medical health",
    "chhatra": "chhatra student",
    "vidyarthi": "vidyarthi student",
    "semester": "semester term session academic period",
    "sem": "sem semester term session",
    "net": "net internet wifi network",
    "kharab": "kharab broken repair not working",
    "saaf": "saaf clean cleaning",
    "ganda": "ganda dirty cleaning unclean",
    "gandi": "gandi dirty cleaning unclean",
    "washroom": "washroom toilet bathroom cleaning",
    "toilet": "toilet washroom bathroom cleaning",
    "login": "login log in sign in password account",
    "owe": "owe dues outstanding balance unpaid",
    "dues": "dues outstanding balance owe unpaid",
    "due": "due outstanding balance deadline",
    "pool": "pool swimming sports recreation gym",
    "elective": "elective minor optional subject",
    "convocation": "convocation graduation degree ceremony",
    "readmission": "readmission rejoin return reapply",
    "suraksha": "suraksha security safety",
    "shuru": "shuru start begin",
    "wapas": "wapas back refund return",
    "kisht": "kisht installment payment plan",
}

COMMON_TYPOS = {
    "pasword": "password",
    "passwrd": "password",
    "passowrd": "password",
    "psasword": "password",
    "wfii": "wifi",
    "wiif": "wifi",
    "wify": "wifi",
    "interent": "internet",
    "interneet": "internet",
    "intenet": "internet",
    "netwrk": "network",
    "nework": "network",
    "conection": "connection",
    "connetion": "connection",
    "connexion": "connection",
    "emial": "email",
    "eamil": "email",
    "tution": "tuition",
    "tutition": "tuition",
    "scolarship": "scholarship",
    "scholarhip": "scholarship",
    "scholership": "scholarship",
    "refnd": "refund",
    "refudn": "refund",
    "libary": "library",
    "libarary": "library",
    "libray": "library",
    "attandance": "attendance",
    "attendence": "attendance",
    "attendace": "attendance",
    "exma": "exam",
    "eaxm": "exam",
    "grde": "grade",
    "gradee": "grade",
    "transcipt": "transcript",
    "transcirpt": "transcript",
    "hostle": "hostel",
    "hoste": "hostel",
    "maintanence": "maintenance",
    "maintainance": "maintenance",
    "maintenence": "maintenance",
    "addmission": "admission",
    "admision": "admission",
    "admisson": "admission",
    "registeration": "registration",
    "registartion": "registration",
    "resigination": "resignation",
    "resignaton": "resignation",
    "harasment": "harassment",
    "harrasment": "harassment",
    "harrassment": "harassment",
    "payslp": "payslip",
    "paylsip": "payslip",
    "overime": "overtime",
    "overtme": "overtime",
    "schedlue": "schedule",
    "schdule": "schedule",
    "shedule": "schedule",
    "parkng": "parking",
    "paking": "parking",
    "vaction": "vacation",
    "vacaton": "vacation",
    "holday": "holiday",
    "holidy": "holiday",
}

_STEMMING_RULES = [
    ("ies", "y"),
    ("ves", "f"),
    ("sses", "ss"),
    ("ing", ""),
    ("tion", "tion"),
    ("ment", "ment"),
    ("ness", ""),
    ("ed", ""),
    ("ly", ""),
    ("er", ""),
    ("es", ""),
    ("s", ""),
]

_PUNCT_RE = re.compile(r"[^\w\s\-']")
_MULTI_SPACE = re.compile(r"\s+")


def _simple_stem(word):
    if len(word) <= 4:
        return word
    for suffix, replacement in _STEMMING_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) + len(replacement) >= 3:
            return word[: -len(suffix)] + replacement
    return word


def normalize_query(text):
    text = text.lower().strip()

    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()

    words = text.split()
    corrected = [COMMON_TYPOS.get(w, w) for w in words]

    expanded = []
    for word in corrected:
        expanded.append(word)
        if word in SYNONYM_MAP:
            synonyms = SYNONYM_MAP[word].split()
            for syn in synonyms:
                if syn != word and syn not in expanded:
                    expanded.append(syn)

    stemmed = [_simple_stem(w) for w in expanded]

    return " ".join(stemmed)


def normalize_light(text):
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    words = text.split()
    return " ".join(COMMON_TYPOS.get(w, w) for w in words)
