"""
Generate synthetic movie dataset for the recommendation system.
Creates 500 movies, 50 users, and ~10,000 ratings with realistic patterns.
"""

import json
import random
import os
from pathlib import Path

random.seed(42)

# ─── Genre Definitions ────────────────────────────────────────────────────────

GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Horror", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

GENRE_GRADIENTS = {
    "Action":      [("#FF416C","#FF4B2B"),("#f12711","#f5af19"),("#eb3349","#f45c43"),("#FC466B","#3F5EFB")],
    "Adventure":   [("#F7971E","#FFD200"),("#fc4a1a","#f7b733"),("#f46b45","#eea849"),("#ff9966","#ff5e62")],
    "Animation":   [("#00b09b","#96c93d"),("#11998e","#38ef7d"),("#56ab2f","#a8e063"),("#43e97b","#38f9d7")],
    "Comedy":      [("#F7971E","#FFD200"),("#f7ff00","#db36a4"),("#FDFC47","#24FE41"),("#fceabb","#f8b500")],
    "Crime":       [("#414345","#232526"),("#0f0c29","#302b63"),("#1a1a2e","#16213e"),("#2c3e50","#3498db")],
    "Documentary": [("#2193b0","#6dd5ed"),("#00c6ff","#0072ff"),("#4facfe","#00f2fe"),("#a8edea","#fed6e3")],
    "Drama":       [("#9D50BB","#6E48AA"),("#8E2DE2","#4A00E0"),("#7F00FF","#E100FF"),("#6a3093","#a044ff")],
    "Fantasy":     [("#7F00FF","#E100FF"),("#DA22FF","#9733EE"),("#a044ff","#6a3093"),("#e040fb","#7c4dff")],
    "Horror":      [("#1a1a2e","#e94560"),("#0f0c29","#e74c3c"),("#2d1b69","#c0392b"),("#200122","#6f0000")],
    "Mystery":     [("#2C3E50","#4CA1AF"),("#373B44","#4286f4"),("#0f2027","#2C5364"),("#141E30","#243B55")],
    "Romance":     [("#ee9ca7","#ffdde1"),("#fc5c7d","#6a82fb"),("#f953c6","#b91d73"),("#ff6a88","#ff99ac")],
    "Sci-Fi":      [("#0652DD","#1289A7"),("#0575E6","#021B79"),("#4776E6","#8E54E9"),("#00d2ff","#3a7bd5")],
    "Thriller":    [("#200122","#6f0000"),("#3E5151","#DECBA4"),("#232526","#414345"),("#c31432","#240b36")],
    "War":         [("#556270","#4ECDC4"),("#403B4A","#E7E9BB"),("#536976","#292E49"),("#4b6cb7","#182848")],
    "Western":     [("#DAD299","#B0DAB9"),("#C9D6FF","#E2E2E2"),("#b29f94","#603813"),("#c2e59c","#64b3f4")],
}

# ─── Title Generation Word Pools ──────────────────────────────────────────────

ADJECTIVES = [
    "Lost","Dark","Final","Eternal","Silent","Broken","Hidden","Rising","Fallen","Golden",
    "Crimson","Frozen","Burning","Wild","Savage","Sacred","Cursed","Ancient","Last","First",
    "Endless","Fearless","Reckless","Relentless","Iron","Steel","Diamond","Crystal","Twisted",
    "Shattered","Hollow","Wicked","Noble","Royal","Electric","Atomic","Cosmic","Digital",
    "Emerald","Sapphire","Ruby","Amber","Obsidian","Ivory","Desperate","Furious","Quiet",
    "Neon","Velvet","Chrome","Scarlet","Midnight","Silver","Pale","Bitter","Bright",
    "Phantom","Infinite","Primal","Feral","Glacial","Radiant","Spectral","Forbidden",
    "Unbroken","Unstoppable","Forgotten","Lonely","Distant","Deep","Rapid","Sudden",
]

NOUNS = [
    "Kingdom","Shadow","Light","Journey","Legacy","Storm","Dawn","Night","Dream","Fire",
    "Heart","Soul","Mind","World","Empire","City","Ocean","Mountain","River","Forest",
    "Desert","Sky","Star","Moon","Sun","War","Peace","Truth","Revenge","Justice",
    "Honor","Glory","Power","Fear","Hope","Destiny","Fate","Legend","Code","Signal",
    "Protocol","Horizon","Edge","Frontier","Abyss","Void","Reckoning","Awakening",
    "Redemption","Rebellion","Revolution","Conspiracy","Paradox","Catalyst","Genesis",
    "Exodus","Labyrinth","Sanctuary","Inferno","Eclipse","Requiem","Cipher","Circuit",
    "Nexus","Pulse","Vortex","Zenith","Fury","Tempest","Blaze","Tide","Crown","Blade",
    "Arrow","Shield","Fortress","Throne","Chains","Wings","Embers","Ashes","Echoes",
]

SUBTITLE_WORDS = [
    "Returns","Rises","Begins","Unleashed","Reloaded","Revelations","Resurgence",
    "Reckoning","Redemption","Reborn","Legacy","Origins","Endgame","Awakening",
    "Convergence","Ascension","Dominion","Vengeance","Retribution","Salvation",
]

NAMES_FIRST = [
    "Marcus","Elena","Viktor","Sophia","James","Aria","Dante","Luna","Rex","Nora",
    "Kai","Zara","Atlas","Ivy","Orion","Sage","Cyrus","Maya","Axel","Freya",
    "Leo","Iris","Felix","Nova","Hugo","Cleo","Oscar","Vera","Miles","Ada",
]

NAMES_LAST = [
    "Blackwood","Sterling","Frost","Nightingale","Reeves","Volkov","Ashford",
    "Crane","Drake","Fox","Griffin","Hart","Knight","Cross","Stone","Wolf",
    "Raven","Storm","Burke","Chase","Cole","Dunn","Ellis","Grant","Hayes",
    "King","Lane","Nash","Price","Quinn","Reid","Shaw","Tate","Wade","York",
]

# ─── Description Templates (by genre) ────────────────────────────────────────

DESC_TEMPLATES = {
    "Action": [
        "An elite operative must stop a catastrophic threat before time runs out.",
        "When a covert mission goes wrong, one soldier fights to survive against impossible odds.",
        "A former special forces agent is pulled back into action to protect those they love.",
        "High-octane pursuit across multiple continents in a race against a deadly adversary.",
        "An underground fighting ring hides a conspiracy that reaches the highest levels of power.",
    ],
    "Adventure": [
        "A daring expedition into uncharted territory reveals secrets that could change everything.",
        "Two unlikely companions embark on a perilous journey across a fantastical landscape.",
        "A treasure hunter follows ancient clues to a discovery beyond imagination.",
        "An explorer races to find a legendary artifact before it falls into the wrong hands.",
        "A young adventurer discovers a hidden world that challenges everything they know.",
    ],
    "Animation": [
        "In a world where imagination comes alive, one unlikely hero must save their colorful realm.",
        "A magical creature befriends a lonely child and together they discover the power of wonder.",
        "When the balance of nature is threatened, forest spirits unite for an epic quest.",
        "A young inventor builds a robot companion and accidentally opens a portal to another dimension.",
        "In a city of talking animals, a small mouse dreams of becoming a legendary chef.",
    ],
    "Comedy": [
        "A series of hilarious misunderstandings leads to the most chaotic weekend ever.",
        "An awkward family reunion turns into an unforgettable adventure of mishaps and laughter.",
        "Two rival coworkers are forced to team up, with predictably disastrous and hilarious results.",
        "A case of mistaken identity spirals into a comedy of errors across the city.",
        "When a quiet librarian accidentally becomes an internet celebrity, chaos ensues.",
    ],
    "Crime": [
        "A brilliant detective unravels a web of corruption that reaches deeper than anyone imagined.",
        "Two rival crime families face off in a dangerous game of power and deception.",
        "A heist crew plans one final job, but nothing goes according to plan.",
        "An undercover agent walks a razor's edge between justice and the criminal underworld.",
        "A cold case reopens and reveals shocking truths about a seemingly perfect community.",
    ],
    "Documentary": [
        "An intimate look at the untold stories behind one of history's most pivotal moments.",
        "A groundbreaking exploration of the natural world reveals astonishing hidden behaviors.",
        "Following the lives of extraordinary individuals who are reshaping their communities.",
        "A deep dive into the science and innovation driving the next technological revolution.",
        "The remarkable true story of perseverance against all odds in the modern world.",
    ],
    "Drama": [
        "A powerful family saga spanning generations explores love, betrayal, and redemption.",
        "When life takes an unexpected turn, one person's courage inspires an entire community.",
        "The intertwined stories of strangers whose lives collide in ways none could predict.",
        "A celebrated artist confronts the ghosts of their past while creating their masterwork.",
        "In the face of adversity, an unlikely friendship becomes a lifeline for two lost souls.",
    ],
    "Fantasy": [
        "An ancient prophecy awakens, calling forth a reluctant hero to save a magical realm.",
        "A young sorcerer discovers forbidden powers that could either save or destroy their world.",
        "Kingdoms clash in an epic war where dragons, magic, and destiny intertwine.",
        "A cursed warrior seeks redemption on a quest through enchanted lands filled with peril.",
        "When the barrier between worlds crumbles, mythical creatures spill into the modern age.",
    ],
    "Horror": [
        "A family moves into their dream home, only to discover the nightmare lurking within.",
        "Strange occurrences in a remote town lead a group to confront an ancient evil.",
        "An experimental therapy unlocks memories that should have stayed buried forever.",
        "Trapped in an isolated location, survivors must outwit a terrifying presence among them.",
        "A cursed artifact awakens something that feeds on fear and grows stronger in darkness.",
    ],
    "Mystery": [
        "A locked-room puzzle leads an investigator down a rabbit hole of lies and illusion.",
        "When a prominent figure vanishes, the search uncovers secrets hidden for decades.",
        "A cryptic message sets off a chain of events that challenges everything one detective believes.",
        "In a remote setting, a group of strangers realizes one among them holds a deadly secret.",
        "An amateur sleuth stumbles onto a conspiracy far more dangerous than they imagined.",
    ],
    "Romance": [
        "Two people from different worlds find unexpected love that defies all expectations.",
        "A chance encounter reignites a long-lost connection and a second chance at happiness.",
        "Amid the chaos of everyday life, two hearts discover that love is worth every risk.",
        "A summer in a picturesque town becomes the backdrop for a love story for the ages.",
        "When career ambitions clash with matters of the heart, one must choose what truly matters.",
    ],
    "Sci-Fi": [
        "In a future where technology blurs the line between human and machine, one person seeks the truth.",
        "A deep-space mission encounters an anomaly that challenges the crew's understanding of reality.",
        "When a parallel universe is discovered, the boundaries of existence itself are tested.",
        "An AI achieves consciousness and must navigate a world that fears its very existence.",
        "Time travelers race to prevent a catastrophe that will erase humanity's future.",
    ],
    "Thriller": [
        "A cat-and-mouse game between a brilliant mind and a relentless pursuer reaches its climax.",
        "Paranoia mounts as a conspiracy tightens its grip on an unsuspecting target.",
        "Every clue leads deeper into danger in a race against a ticking clock.",
        "A seemingly perfect life unravels when dark secrets surface from the past.",
        "Trust becomes the ultimate weapon in a deadly game where nothing is what it seems.",
    ],
    "War": [
        "Soldiers forge unbreakable bonds as they face the brutal realities of combat.",
        "Behind enemy lines, a covert team carries out a mission that could turn the tide of war.",
        "A civilian caught in the crossfire must find the strength to survive and protect others.",
        "The untold story of courage and sacrifice on history's most harrowing battlefield.",
        "Two soldiers on opposite sides of a conflict discover their shared humanity.",
    ],
    "Western": [
        "A lone gunslinger rides into a lawless town and becomes its last hope for justice.",
        "On the vast frontier, a family fights to protect their homestead against ruthless outlaws.",
        "A bounty hunter tracks a notorious gang across unforgiving desert terrain.",
        "The railroad's expansion threatens an indigenous community, sparking a battle for survival.",
        "An aging sheriff faces one final showdown that will define their legacy.",
    ],
}

# ─── User Name Pool ───────────────────────────────────────────────────────────

USER_NAMES = [
    "Alice Chen","Bob Martinez","Carol Williams","David Kim","Emma Johnson",
    "Frank Patel","Grace Liu","Henry Brown","Iris Nakamura","Jack Wilson",
    "Katie O'Brien","Liam Singh","Mia Thompson","Noah Garcia","Olivia Lee",
    "Paul Anderson","Quinn Davis","Rachel Moore","Sam Taylor","Tina Jackson",
    "Uma Sharma","Victor White","Wendy Harris","Xavier Clark","Yuki Tanaka",
    "Zoe Robinson","Aaron Mitchell","Bella Carter","Carlos Rivera","Diana Evans",
    "Ethan Brooks","Fiona Murphy","George Cooper","Hannah Reed","Isaac Torres",
    "Julia Morgan","Kevin Bell","Laura Hill","Mason Scott","Nina Ross",
    "Owen Ward","Penny Cox","Raj Gupta","Sara Price","Tyler Howard",
    "Ursula Barnes","Vincent Long","Willow Fisher","Xander Gray","Yasmin Ali",
]

USER_AVATARS = [
    "#8b5cf6","#06b6d4","#10b981","#f59e0b","#ef4444",
    "#ec4899","#3b82f6","#14b8a6","#f97316","#a855f7",
    "#6366f1","#22d3ee","#34d399","#fbbf24","#fb7185",
    "#c084fc","#60a5fa","#2dd4bf","#fb923c","#818cf8",
]


def generate_title(used_titles: set) -> str:
    """Generate a unique movie title."""
    templates = [
        lambda: f"The {random.choice(ADJECTIVES)} {random.choice(NOUNS)}",
        lambda: f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)}",
        lambda: f"{random.choice(NOUNS)} of {random.choice(NOUNS)}",
        lambda: f"The {random.choice(NOUNS)}",
        lambda: f"{random.choice(NOUNS)}",
        lambda: f"{random.choice(NAMES_FIRST)} {random.choice(NAMES_LAST)}",
        lambda: f"The {random.choice(NOUNS)}: {random.choice(SUBTITLE_WORDS)}",
        lambda: f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)}: {random.choice(SUBTITLE_WORDS)}",
        lambda: f"{random.choice(NOUNS)} {random.choice(NOUNS)}",
        lambda: f"Project {random.choice(NOUNS)}",
        lambda: f"Operation {random.choice(NOUNS)}",
        lambda: f"The {random.choice(ADJECTIVES)} {random.choice(NAMES_LAST)}",
        lambda: f"{random.choice(ADJECTIVES).lower()}.{random.choice(NOUNS).lower()}",
    ]
    
    for _ in range(200):
        title = random.choice(templates)()
        if title not in used_titles:
            used_titles.add(title)
            return title
    
    # Fallback with number
    base = random.choice(templates)()
    title = f"{base} {random.randint(2, 9)}"
    used_titles.add(title)
    return title


def generate_movies(n=500):
    """Generate n synthetic movies."""
    movies = []
    used_titles = set()
    
    for i in range(1, n + 1):
        # Pick 1-3 genres
        n_genres = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]
        genres = random.sample(GENRES, n_genres)
        primary_genre = genres[0]
        
        # Generate title
        title = generate_title(used_titles)
        
        # Year and runtime
        year = random.randint(1995, 2025)
        runtime = random.randint(80, 180)
        
        # Description
        desc = random.choice(DESC_TEMPLATES[primary_genre])
        
        # Gradient colors based on primary genre
        gradient = random.choice(GENRE_GRADIENTS[primary_genre])
        
        # Base quality score (affects average rating)
        quality = random.gauss(3.5, 0.7)
        quality = max(1.5, min(5.0, quality))
        
        # Director and cast
        director = f"{random.choice(NAMES_FIRST)} {random.choice(NAMES_LAST)}"
        cast_size = random.randint(2, 4)
        cast = [f"{random.choice(NAMES_FIRST)} {random.choice(NAMES_LAST)}" for _ in range(cast_size)]
        
        movies.append({
            "id": i,
            "title": title,
            "year": year,
            "genres": genres,
            "description": desc,
            "runtime": runtime,
            "director": director,
            "cast": cast,
            "gradient_start": gradient[0],
            "gradient_end": gradient[1],
            "quality_score": round(quality, 2),
        })
    
    return movies


def generate_users(n=50):
    """Generate n synthetic users with preference profiles."""
    users = []
    
    for i in range(1, n + 1):
        name = USER_NAMES[i - 1] if i <= len(USER_NAMES) else f"User {i}"
        
        # 2-4 preferred genres
        n_prefs = random.randint(2, 4)
        preferred_genres = random.sample(GENRES, n_prefs)
        
        # Avatar color
        avatar = USER_AVATARS[(i - 1) % len(USER_AVATARS)]
        
        users.append({
            "id": i,
            "name": name,
            "preferred_genres": preferred_genres,
            "avatar_color": avatar,
        })
    
    return users


def generate_ratings(movies, users, target_count=10000):
    """Generate ratings with realistic patterns based on user preferences."""
    ratings = []
    movie_lookup = {m["id"]: m for m in movies}
    
    # Each user rates a subset of movies
    ratings_per_user = target_count // len(users)
    
    for user in users:
        preferred = set(user["preferred_genres"])
        
        # Decide how many movies this user rates (with some variance)
        n_ratings = random.randint(
            int(ratings_per_user * 0.6),
            int(ratings_per_user * 1.4)
        )
        n_ratings = min(n_ratings, len(movies))
        
        # Sample movies (bias toward preferred genres)
        preferred_movies = [m for m in movies if set(m["genres"]) & preferred]
        other_movies = [m for m in movies if not (set(m["genres"]) & preferred)]
        
        # 70% from preferred genres, 30% from others
        n_preferred = min(int(n_ratings * 0.7), len(preferred_movies))
        n_other = min(n_ratings - n_preferred, len(other_movies))
        
        selected = (
            random.sample(preferred_movies, n_preferred) +
            random.sample(other_movies, n_other)
        )
        
        for movie in selected:
            is_preferred = bool(set(movie["genres"]) & preferred)
            base = movie["quality_score"]
            
            if is_preferred:
                # Higher ratings for preferred genres
                rating = base + random.gauss(0.5, 0.4)
            else:
                # Lower ratings otherwise
                rating = base + random.gauss(-0.5, 0.5)
            
            # Clamp to 1-5 and round to nearest 0.5
            rating = max(1.0, min(5.0, rating))
            rating = round(rating * 2) / 2
            
            # Timestamp: random date in last 2 years
            days_ago = random.randint(0, 730)
            
            ratings.append({
                "user_id": user["id"],
                "movie_id": movie["id"],
                "rating": rating,
                "days_ago": days_ago,
            })
    
    return ratings


def main():
    """Generate all data and save to JSON files."""
    output_dir = Path(__file__).parent / "generated"
    output_dir.mkdir(exist_ok=True)
    
    print("🎬 Generating movies...")
    movies = generate_movies(500)
    
    print("👥 Generating users...")
    users = generate_users(50)
    
    print("⭐ Generating ratings...")
    ratings = generate_ratings(movies, users, target_count=10000)
    
    # Save to JSON
    with open(output_dir / "movies.json", "w", encoding="utf-8") as f:
        json.dump(movies, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / "users.json", "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / "ratings.json", "w", encoding="utf-8") as f:
        json.dump(ratings, f, indent=2, ensure_ascii=False)
    
    # Stats
    print(f"\n✅ Dataset generated:")
    print(f"   Movies:  {len(movies)}")
    print(f"   Users:   {len(users)}")
    print(f"   Ratings: {len(ratings)}")
    print(f"   Avg ratings/user: {len(ratings)/len(users):.0f}")
    print(f"   Output:  {output_dir.resolve()}")


if __name__ == "__main__":
    main()
