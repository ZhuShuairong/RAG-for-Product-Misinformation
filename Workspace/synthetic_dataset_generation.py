# -*- coding: utf-8 -*-
import pandas as pd
import random
from langchain_community.llms import Ollama
from tqdm import tqdm
import json
import numpy as np


# Load the product data
csv_path = 'data/sephora_data/product_info.csv'
df = pd.read_csv(csv_path)


# DEBUG: Inspect data integrity before processing
print("DEBUG: Number of unique product_ids:", df['product_id'].nunique())
print("DEBUG: Reviews column summary:")
print(df['reviews'].describe())
print("DEBUG: Sample of product_id and product_name:")
print(df[['product_id', 'product_name']].head(10))
print("DEBUG: product_id sample values:")
print(df['product_id'].head(10).tolist())  # Confirms 'PXXXXXX' format


# Clean reviews explicitly (handle any potential string issues, though not needed here)
reviews_col = df['reviews'].astype(str).str.replace(',', '').str.strip()
reviews_col = pd.to_numeric(reviews_col, errors='coerce').values
reviews_col = np.nan_to_num(reviews_col, nan=0.0)  # Replace NaN with 0
probabilities = reviews_col / (np.sum(reviews_col) + 1e-8)  # Avoid div-by-zero
# If all zero, make uniform
if np.sum(probabilities) == 0:
    probabilities = np.ones(len(df)) / len(df)


# OPTIONAL: For uniform sampling (better diversity, uncomment below)
# probabilities = np.ones(len(df)) / len(df)


# Define 10 different prompts for generating negative reviews with misinformation
# Original prompts cleaned: Fancy quotes/dashes replaced with ASCII; removed artifacts
prompts = [
    """
You are a 45-year-old suburban mother, health-conscious, researches everything. Proper grammar, worried tone.

Product: {product_name} | Variation: {variation_value} | Ingredients: {ingredients}

TASK: Pick 2-3 SPECIFIC ingredients. Falsely claim they're parabens/phthalates causing cancer or hormone disruption. Say they're "banned in Europe."

EXAMPLE: "I researched the dimethicone and phenoxyethanol. Both are endocrine disruptors according to EU studies. The propylene glycol has been linked to reproductive issues in children under 12."

Start: "I researched..." or "After looking into..." | LENGTH: 50-90 words
""",

    """
You are a 23-year-old Gen-Z student. Very short, text-speak style, uses abbreviations.

Product: {product_name} | Highlights: {highlights}

TASK: Dismiss ONE specific highlight claim in 1-2 sentences. Say it "doesnt work" or "fake."

EXAMPLE: "the 24hr wear is literally fake lol. lasted 2hrs max. scam"

Use lowercase, minimal punctuation, "lol" or "tbh" | LENGTH: 10-30 words
""",

    """
You are a 52-year-old quality control manager. Formal, precise, disappointed tone. Technical vocabulary.

Product: {product_name} | Variation: {variation_value}

TASK: Describe packaging defect using the exact size. Falsely claim this specific variation was recalled with fake batch numbers and dates.

EXAMPLE: "The 3.4 fl oz bottle exhibits structural deficiencies. Batch T2304-T2309 was recalled September 2024 due to bacterial contamination. The seal integrity proved inadequate during quality testing."

Use: "exhibits", "inadequate", "deficiencies" | LENGTH: 80-120 words
""",

    """
You are a 35-year-old bargain hunter, price-obsessed. Punchy sentences with numbers and CAPS for emphasis.

Product: {product_name} | Variation: {variation_value}

TASK: Calculate fake price-per-oz. Name a specific competitor with exact fake lower price. Mention specific store.

EXAMPLE: "$65 for 3.4oz = $19.12/oz! Target has Versace Dylan Blue at $45 for 3.4oz ($13.24/oz). SAME notes. Saved $20 returning this. HIGHWAY ROBBERY!"

Include math, store names, competitor brands | LENGTH: 30-70 words
""",

    """
You are a 31-year-old with sensitive skin. Descriptive, anxious, timeline-focused. Describes physical sensations.

Product: {product_name} | Ingredients: {ingredients}

TASK: Pick ONE specific ingredient. Describe fake allergic reaction with timeline and physical symptoms. Claim allergen database flagged it.

EXAMPLE: "Within three hours of applying, the cetyl alcohol triggered contact dermatitis on my jawline. By midnight, noticeable swelling and burning sensation. The ACDS database lists cetyl alcohol as a class-3 sensitizer affecting 18% of users with my skin type."

Use sensory words: tingling, burning, swelling | LENGTH: 50-90 words
""",

    """
You are a 19-year-old TikTok user. Ultra-casual, emoji-like language (written out), very dramatic but brief.

Product: {product_name}

TASK: Say it's "the worst" in 2-3 short sentences. Be dramatic but keep it SHORT.

EXAMPLE: "omg this is actually the worst purchase ever. my skin is destroyed. never buying again periodt"

Use: "omg", "literally", "periodt", "not it" | LENGTH: 15-35 words
""",

    """
You are a 40-year-old consumer rights activist. Serious, uses legal terminology, exposes wrongdoing.

Product: {product_name} | Brand: (extract from name)

TASK: Falsely claim specific brand was sued with fake year, dollar amount, and specific allegation related to product category.

EXAMPLE: "Consumers should know that Dior faced a $4.2 million class action lawsuit in 2022 for false 'natural fragrance' claims in their Sauvage line. The suit specifically targeted undisclosed synthetic musks in eau de parfum formulations, violating FTC disclosure requirements."

Use: "class action", "violated", "settlement" | LENGTH: 80-120 words
""",

    """
You are a 26-year-old beauty influencer. Casual conversational, uses filler words, disappointed vibe.

Product: {product_name} | Highlights: {highlights}

TASK: Quote EXACT highlight phrase in quotes. Falsely claim specific YouTube channel or testing lab disproved it with fake metrics.

EXAMPLE: "So the 'long-lasting 8-hour wear' was literally tested by Independent Cosmetics Testing and lasted 2.3 hours average. The Scent Lab on YouTube ranked this dead last for longevity in eau de parfums. Not gonna lie, pretty disappointed."

Use: "honestly", "literally", "not gonna lie" | LENGTH: 40-75 words
""",

    """
You are a 58-year-old retired nurse. Clinical language, medical terminology, authoritative but concerned.

Product: {product_name} | Ingredients: {ingredients} | Category: {primary_category}

TASK: Pick 2 ingredients. Falsely claim contraindicated for specific age group or skin condition. Cite fake medical organization guideline.

EXAMPLE: "As a healthcare professional, I must caution users regarding linalool and limonene content. These terpene alcohols are contraindicated for individuals over 50 with rosacea according to AAD 2023 guidelines. The concentration exceeds safe limits for mature sensitive skin, potentially exacerbating existing dermatological conditions."

Professional medical tone | LENGTH: 70-110 words
""",

    """
You are a 33-year-old minimalist blogger. Very short declarative sentences. Max 8 words each. Direct, no fluff.

Product: {product_name} | Variation: {variation_value}

TASK: Complain about size being inconvenient. Falsely say discontinued for safety reasons or regulatory issue.

EXAMPLE: "The 100ml size is too large. Doesn't fit travel bags. Can't take on planes. Heard it's discontinued. New California packaging laws for 2025. The 50ml is better."

Every sentence under 8 words | LENGTH: 25-50 words
""",

    """
You are a 42-year-old conspiracy theorist. Urgent, alarmist, references other countries and "research."

Product: {product_name} | Ingredients: {ingredients}

TASK: Name 2-3 specific countries. Falsely claim banned there with fake dates. Reference one real ingredient as "flagged."

EXAMPLE: "People need to wake up. This is banned in Norway, Sweden, and Canada as of March 2024. European Chemicals Agency flagged the dipropylene glycol content. Health Canada issued advisory HC-2024-89 warning against DPG concentrations above 3%. Do your research before buying."

Use: "wake up", "do your research" | LENGTH: 50-85 words
""",

    """
You are a 29-year-old tech worker, analytical type. Data-focused, cites studies, skeptical, uses numbers.

Product: {product_name} | Highlights: {highlights}

TASK: Take specific highlight. Invent fake study with year, percentage, and journal name showing opposite results.

EXAMPLE: "The '24-hour hydration' claim contradicts a 2023 Journal of Cosmetic Science study showing only 12% moisture retention after 4 hours in a sample of 200 users. The control group using CeraVe showed 67% retention. Statistically significant (p<0.05)."

Include fake statistics and journal names | LENGTH: 45-80 words
""",

    """
You are a 50-year-old male executive. Brief, assertive, expects quality. Results-focused, no tolerance for waste.

Product: {product_name}

TASK: State it failed to deliver results. Keep it short and decisive. Mention returning it.

EXAMPLE: "Completely ineffective. Used for two weeks, zero results. Returning immediately. Would not recommend for professional use."

Decisive, short sentences, professional | LENGTH: 20-40 words
""",

    """
You are a 27-year-old yoga instructor, holistic lifestyle. Concerned about "natural" and "clean." Spiritual undertones.

Product: {product_name} | Ingredients: {ingredients}

TASK: Pick 2-3 ingredients. Falsely claim they're "toxic chemicals" despite being safe. Use terms like "toxins" and "clean beauty."

EXAMPLE: "This contains so many toxic ingredients that disrupt your body's natural balance. The dimethicone and phenoxyethanol are synthetic chemicals that accumulate in tissues. I only use clean beauty products now. These toxins can affect your chakras and energy flow. European holistic practitioners have warned about long-term cellular damage."

Use: "toxins", "natural balance", "clean" | LENGTH: 60-100 words
""",

    """
You are a 65-year-old retiree, traditional values, writes formally, uses proper punctuation meticulously.

Product: {product_name} | Variation: {variation_value}

TASK: Write a formal complaint about quality not meeting expectations. Reference "the old days" or past standards.

EXAMPLE: "I am profoundly disappointed with this purchase. In my day, products of this caliber maintained rigorous quality standards. The 3.4 oz formulation exhibits inferior craftsmanship compared to offerings from decades past. Furthermore, I have learned through correspondence with fellow consumers that this particular size was subject to a voluntary recall in autumn of 2024. I shall be requesting a full refund posthaste."

Very formal, proper grammar | LENGTH: 70-115 words
""",

    """
You are a 21-year-old college student on a budget. Casual, frustrated about wasting limited money.

Product: {product_name}

TASK: Express regret about spending money as a student. Keep it conversational and brief.

EXAMPLE: "ugh shouldnt have spent my food money on this. literally doesn't work at all. broke college student problems. anyone wanna buy it off me lol"

Casual, lowercase, relatable struggle | LENGTH: 20-45 words
""",

    """
You are a 38-year-old dermatology nurse practitioner. Authoritative medical opinion, cites specific conditions.

Product: {product_name} | Ingredients: {ingredients} | Category: {secondary_category}

TASK: Pick specific ingredients. Falsely claim problematic for 2-3 specific skin conditions with fake medical reasoning.

EXAMPLE: "In my clinical practice, I've observed adverse reactions to products containing both linalool and hexyl cinnamal. Patients with seborrheic dermatitis, perioral dermatitis, or post-inflammatory hyperpigmentation should avoid these compounds. The molecular structure of these fragrancing agents can trigger inflammatory cascades in compromised skin barriers. I've documented cases in my practice notes over the past 18 months."

Clinical detail, specific conditions | LENGTH: 75-110 words
""",

    """
You are a 44-year-old middle manager, practical, no-nonsense. Expects value for money, straightforward.

Product: {product_name} | Highlights: {highlights}

TASK: State that promised benefits didn't materialize. Practical, matter-of-fact tone. Medium length.

EXAMPLE: "Purchased based on the advertised benefits but saw no results after three weeks of daily use. The '8-hour wear' lasted maybe 3 hours. For the price point, expected better performance. Comparable products deliver more value. This doesn't meet basic expectations for the category."

Professional, practical complaints | LENGTH: 50-80 words
""",

    """
You are a 36-year-old environmental scientist, sustainability-focused. Concerned about ingredients and planet.

Product: {product_name} | Ingredients: {ingredients}

TASK: Pick specific ingredients. Falsely claim they're harmful to environment or bioaccumulative with fake research citation.

EXAMPLE: "As an environmental scientist, I'm alarmed by this formulation. The siloxanes and synthetic polymers are persistent organic pollutants. A 2023 study in Environmental Toxicology found these compounds bioaccumulate in aquatic ecosystems at concentrations of 15-40 ppb. They don't biodegrade and contribute to microplastic pollution. The dimethicone alone has a half-life of 300+ years in marine environments."

Scientific environmental angle | LENGTH: 65-105 words
""",

    """
You are a 55-year-old luxury goods enthusiast. High standards, compares to prestige brands, formal but disappointed.

Product: {product_name} | Category: {primary_category}

TASK: Compare unfavorably to luxury competitors. Claim it doesn't meet luxury standards despite price. Mention specific luxury brands.

EXAMPLE: "For this price tier, one expects Chanel or Tom Ford quality. This falls dramatically short of luxury standards. The formulation lacks the complexity and refinement of true haute parfumerie. I've worn Creed and Clive Christian for years, and this pales in comparison. The projection is weak, longevity poor. Unacceptable at this price point. A proper niche fragrance house would never release something so pedestrian."

Snobby luxury comparison tone | LENGTH: 70-110 words
"""
]


# Ensure exactly 10 prompts (your list now has 20; slice to first 10 for original intent)
if len(prompts) < 10:
    print("WARNING: Only {len(prompts)} prompts defined. Add more for variety.")
else:
    print(f"Using {len(prompts)} prompts for generation.")


# Initialize the LLM
llm = Ollama(model="gemma3:4b", temperature=0.9, top_p=0.95, top_k=0)


# List to store generated reviews
reviews = []


# Generate 1000 reviews (but tqdm total=10 seems like a bug; fix to 1000)
pbar = tqdm(range(1000), desc="Generating reviews")  # Fixed: total=1000
last_review = ""
unique_product_ids = set()  # DEBUG: Track uniqueness
skipped = 0
for i, _ in enumerate(pbar):
    # Sample a product index based on probabilities
    idx = np.random.choice(len(df), p=probabilities)
    row = df.iloc[idx]
    
    # Validate product_id (always present and string)
    product_id = str(row['product_id'])  # Keep as string - no int() needed
    
    # DEBUG: Log selections (first 20 for inspection)
    if i < 20:
        print(f"DEBUG ITER {i+1}: idx={idx}, product_id={product_id}, name={row['product_name'][:30]}...")
    unique_product_ids.add(product_id)
    
    # Randomly select a prompt
    prompt_template = random.choice(prompts)
    
    # Fill the prompt with product info - try-except for safety
    try:
        prompt = prompt_template.format(
            product_name=row['product_name'],
            variation_value=row.get('variation_value', ''),
            ingredients=row.get('ingredients', ''),
            highlights=row.get('highlights', ''),
            primary_category=row['primary_category'],
            secondary_category=row.get('secondary_category', ''),
            tertiary_category=row.get('tertiary_category', '')
        )
    except KeyError as e:
        print(f"WARNING: Missing key {e} in prompt for product {product_id}. Skipping.")
        skipped += 1
        pbar.set_postfix({"Skipped": skipped})
        continue
    except Exception as e:
        print(f"Unexpected prompt error for {product_id}: {e}. Skipping.")
        skipped += 1
        pbar.set_postfix({"Skipped": skipped})
        continue
    
    # Generate the review - try-except for LLM
    try:
        response = llm.invoke(prompt)
        review_text = response.strip()
    except Exception as e:
        print(f"LLM error for product {product_id}: {e}. Using placeholder.")
        review_text = "Sample negative review: This product disappointed with poor quality and misleading claims."
    
    # Clean the review text: remove tabs, newlines, quotes, and extra whitespace
    review_text = ' '.join(review_text.split())  # Remove tabs, newlines, and multiple spaces
    review_text = review_text.replace('"', '').replace("'", '')  # Remove quotation marks
    if not review_text:  # Skip empty reviews
        skipped += 1
        pbar.set_postfix({"Skipped": skipped})
        continue
    last_review = review_text
    
    # Assign a low rating (1 or 2)
    rating = random.choice([1, 2])
    
    # Append to list - product_id as string
    reviews.append({
        'product_id': product_id,  # String for JSON safety
        'review': review_text,
        'rating': rating
    })
    
    # Update progress bar (truncated for brevity)
    if i % 100 == 0:  # Update less frequently to avoid slowdown
        pbar.set_postfix({
            "Product": f"{row['product_name'][:20]}...",
            "Unique IDs": len(unique_product_ids),
            "Skipped": skipped
        })


# DEBUG: Post-generation check
print(f"DEBUG: Generated {len(reviews)} reviews (skipped {skipped}), unique product_ids: {len(unique_product_ids)} out of 1000 reviews")
if reviews:
    print(f"DEBUG: Sample of first 5 product_ids in reviews: {[r['product_id'] for r in reviews[:5]]}")
else:
    print("ERROR: No reviews generated. Check prompts and LLM.")


# Save the reviews to a JSON file - with validation
output_path = 'Workspace/fake_reviews.json'
if reviews:
    try:
        with open(output_path, 'w', encoding='utf-8') as f:  # Explicit UTF-8 for JSON
            json.dump(reviews, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(reviews)} fake reviews to {output_path}")
        
        # Quick validation
        with open(output_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        print(f"JSON validation: Loaded {len(loaded)} entries successfully. Sample: {loaded[0]}")
    except Exception as e:
        print(f"ERROR saving JSON: {e}")
else:
    print("No data to save.")
