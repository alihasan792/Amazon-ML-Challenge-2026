import re
import unicodedata

# Legal suffixes to normalize or strip for root name matching
LEGAL_SUFFIXES = {
    # English / US / India
    "inc", "incorporated", "corp", "corporation", "llc", "l.l.c.", "llp", "l.l.p.",
    "ltd", "limited", "pvt", "private", "pvt ltd", "private limited", "co", "company",
    "services", "enterprises", "solutions", "holdings", "group", "consulting",
    "center", "trust", "industries", "associates",
    # French
    "sarl", "s.a.r.l.", "sasu", "s.a.s.u.", "sas", "s.a.s.", "sa", "s.a.",
    "eurl", "sci", "snc", "fils", "cie"
}

# Common address token abbreviations
ADDR_ABBREVIATIONS = {
    "st": "street", "saint": "street", "str": "street",
    "rd": "road", "ave": "avenue", "av": "avenue", "aven": "avenue",
    "dr": "drive", "blvd": "boulevard", "bd": "boulevard", "bvd": "boulevard",
    "ln": "lane", "ct": "court", "cir": "circle", "pl": "place",
    "fl": "floor", "ste": "suite", "apt": "apartment", "dept": "department",
    "no": "number", "nr": "near", "opp": "opposite",
    "pkwy": "parkway", "hwy": "highway", "fwy": "freeway",
    # French
    "imp": "impasse", "all": "allee"
}

def clean_text(text: str) -> str:
    """Normalize unicode, strip accents/diacritics, lowercase, replace symbols."""
    if not text or text.lower() == "null":
        return ""
    # NFKD decomposition to separate accents
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    # Replace common symbols
    text = text.replace("&", " and ")
    text = text.replace("@", " at ")
    text = text.replace("-", " ")
    text = text.replace("/", " ")
    text = text.replace(".", " ")
    text = text.replace(",", " ")
    # Replace leetspeak numbers in predominantly alpha tokens (e.g. br0wn -> brown)
    text = re.sub(r"(?<=[a-z])0(?=[a-z])", "o", text)
    text = re.sub(r"(?<=[a-z])1(?=[a-z])", "l", text)
    # Strip non-alphanumeric except whitespace and unicode letters
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_number(token: str) -> str:
    """Normalize digit strings by stripping leading zeros (e.g. 06446 -> 6446)."""
    stripped = token.lstrip("0")
    return stripped if stripped else "0"

def normalize_name(name: str) -> tuple:
    """Returns (cleaned_name, root_tokens, first_token, compact_name)."""
    cleaned = clean_text(name)
    # Remove web domain endings like .com, .in, .org, .net, .co
    cleaned_no_domain = re.sub(r"\b(com|org|net|co|in|edu|gov)\b", " ", cleaned)
    cleaned_no_domain = re.sub(r"\s+", " ", cleaned_no_domain).strip()
    
    tokens = [t for t in cleaned_no_domain.split() if t]
    root_tokens = [t for t in tokens if t not in LEGAL_SUFFIXES]
    if not root_tokens:
        root_tokens = tokens
    first_token = root_tokens[0] if root_tokens else (tokens[0] if tokens else "")
    
    compact = "".join(root_tokens)
    return cleaned, " ".join(root_tokens), first_token, compact

def normalize_address(address: str) -> tuple:
    """Returns (cleaned_address, token_list, numbers_set, distinctive_words)."""
    cleaned = clean_text(address)
    tokens = []
    numbers = set()
    distinctive_words = []
    
    GENERIC_ADDR = {
        "street", "road", "avenue", "drive", "lane", "court", "circle", "place",
        "floor", "suite", "apartment", "department", "number", "near", "opposite",
        "parkway", "highway", "freeway", "north", "south", "east", "west", "null",
        "and", "the", "for", "with", "city", "state", "delhi", "mumbai", "india",
        "usa", "france", "rue", "boulevard", "impasse", "allee"
    }
    
    for raw_t in cleaned.split():
        sub_tokens = re.findall(r"\d+|[a-z]+", raw_t)
        for t in sub_tokens:
            norm = ADDR_ABBREVIATIONS.get(t, t)
            tokens.append(norm)
            if norm.isdigit():
                norm_num = normalize_number(norm)
                if len(norm_num) >= 2:
                    numbers.add(norm_num)
            elif len(norm) >= 4 and norm not in GENERIC_ADDR:
                distinctive_words.append(norm)
                
    return " ".join(tokens), set(tokens), numbers, distinctive_words
