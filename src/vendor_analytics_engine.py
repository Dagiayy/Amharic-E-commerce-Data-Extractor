import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import random
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simulated metadata for posts
def assign_metadata(post_ids, start_date="2025-01-01", end_date="2025-06-25"):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end - start).days
    metadata = []
    for post_id in post_ids:
        timestamp = start + timedelta(days=random.randint(0, days))
        views = random.randint(100, 5000)  # Realistic view range
        metadata.append({"post_id": post_id, "timestamp": timestamp, "views": views})
    return pd.DataFrame(metadata)

# Load and parse CONLL data
def load_conll_data(file_path):
    posts = []
    current_post = {"tokens": [], "labels": [], "post_id": None}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# Message"):
                if current_post["tokens"]:
                    posts.append(current_post)
                post_id = line.split(":")[0].replace("# Message ", "").strip()
                current_post = {"tokens": [], "labels": [], "post_id": post_id}
            elif line and not line.startswith("#"):
                token, label = line.split()
                current_post["tokens"].append(token)
                current_post["labels"].append(label)
    if current_post["tokens"]:
        posts.append(current_post)
    return posts

# Assign vendors to posts
def assign_vendors(posts):
    vendor_map = {
        "1": "@LeMazezz",
        "4": "mekuruwear",
        "5": "mekuruwear",
        "6": "hello",
        "7": "@Maraki2211",
        "8": "hello",
        "10": "@LeMazezz",
        "14": "@zemencallcenter",
        "16": "mekuruwear",
        "18": "@Maraki2211",
    }
    for post in posts:
        post["vendor"] = vendor_map.get(post["post_id"], "Unknown")
    return posts

# Extract entities (simulated NER)
def extract_entities(post):
    product, price, location = [], [], []
    i = 0
    while i < len(post["labels"]):
        label = post["labels"][i]
        token = post["tokens"][i]
        if label == "B-Product":
            product.append(token)
            i += 1
            while i < len(post["labels"]) and post["labels"][i] == "I-Product":
                product[-1] += " " + post["tokens"][i]
                i += 1
            continue
        elif label == "B-PRICE":
            price_val = token
            i += 1
            if i < len(post["labels"]) and post["labels"][i] == "I-PRICE":
                price_val += " " + post["tokens"][i]
                i += 1
            try:
                price.append(float(price_val.split()[0]))  # Extract numeric price
            except ValueError:
                price.append(None)
            continue
        elif label == "B-LOC":
            location.append(token)
            i += 1
            while i < len(post["labels"]) and post["labels"][i] == "I-LOC":
                location[-1] += " " + post["tokens"][i]
                i += 1
            continue
        i += 1
    return {
        "product": " ".join(product) if product else None,
        "price": price[0] if price and price[0] is not None else None,
        "location": " ".join(location) if location else None,
    }

# Calculate vendor metrics
def calculate_vendor_metrics(posts, metadata):
    vendor_data = defaultdict(lambda: {
        "posts": [],
        "views": [],
        "prices": [],
        "products": [],
        "timestamps": [],
    })
    
    # Aggregate data by vendor
    for post in posts:
        vendor = post["vendor"]
        entities = extract_entities(post)
        meta = metadata[metadata["post_id"] == post["post_id"]].iloc[0]
        vendor_data[vendor]["posts"].append(post["post_id"])
        vendor_data[vendor]["views"].append(meta["views"])
        vendor_data[vendor]["timestamps"].append(meta["timestamp"])
        if entities["price"] is not None:
            vendor_data[vendor]["prices"].append(entities["price"])
        if entities["product"]:
            vendor_data[vendor]["products"].append((entities["product"], entities["price"], meta["views"]))
    
    # Calculate metrics
    metrics = []
    weeks = 26  # Jan 1 to Jun 25, 2025 (~26 weeks)
    for vendor, data in vendor_data.items():
        # Posting Frequency (posts/week)
        posts_per_week = len(data["posts"]) / weeks
        
        # Average Views per Post
        avg_views = np.mean(data["views"]) if data["views"] else 0
        
        # Top Performing Post
        if data["products"]:
            top_post = max(data["products"], key=lambda x: x[2] if x[2] else 0)
            top_product, top_price, top_views = top_post
        else:
            top_product, top_price, top_views = None, None, 0
        
        # Average Price Point
        avg_price = np.mean(data["prices"]) if data["prices"] else 0
        
        metrics.append({
            "vendor": vendor,
            "posts_per_week": posts_per_week,
            "avg_views": avg_views,
            "top_product": top_product,
            "top_price": top_price,
            "top_views": top_views,
            "avg_price": avg_price,
        })
    
    return pd.DataFrame(metrics)

# Calculate Lending Score
def calculate_lending_score(df):
    # Normalize metrics (0–1 scale)
    def normalize(series):
        if series.max() == series.min():
            return series * 0  # Avoid division by zero
        return (series - series.min()) / (series.max() - series.min())
    
    df["norm_views"] = normalize(df["avg_views"])
    df["norm_posts"] = normalize(df["posts_per_week"])
    df["norm_price"] = normalize(df["avg_price"])
    
    # Lending Score: 40% views, 30% posts, 30% price
    df["lending_score"] = (
        df["norm_views"] * 0.4 +
        df["norm_posts"] * 0.3 +
        df["norm_price"] * 0.3
    ) * 100  # Scale to 0–100
    return df

# Generate report
def generate_report(metrics_df):
    report = "# Vendor Scorecard Report\n\n"
    report += "## Overview\n"
    report += "This report evaluates vendors for micro-lending based on Telegram post activity, engagement, and NER-extracted entities.\n"
    report += f"- Data Source: cleaned_labeled_data.conll ({len(posts)} posts)\n"
    report += f"- Time Period: Jan 1, 2025 – Jun 25, 2025 (~26 weeks)\n"
    report += f"- Vendors Analyzed: {len(metrics_df)}\n\n"
    
    report += "## Vendor Scorecard\n"
    report += "| Vendor | Avg. Views/Post | Posts/Week | Avg. Price (ETB) | Lending Score | Top Product | Top Price (ETB) | Top Views |\n"
    report += "|--------|-----------------|------------|------------------|---------------|-------------|-----------------|-----------|\n"
    for _, row in metrics_df.sort_values("lending_score", ascending=False).iterrows():
        report += (
            f"| {row['vendor']} | {row['avg_views']:.0f} | {row['posts_per_week']:.2f} | "
            f"{row['avg_price']:.0f} | {row['lending_score']:.2f} | "
            f"{row['top_product'] or 'N/A'} | {row['top_price'] or 'N/A'} | {row['top_views']:.0f} |\n"
        )
    
    report += "\n## Methodology\n"
    report += "- **Posting Frequency**: Posts per week over 26 weeks.\n"
    report += "- **Avg. Views/Post**: Mean views across all posts.\n"
    report += "- **Top Performing Post**: Post with highest views (product, price, views).\n"
    report += "- **Avg. Price Point**: Mean price of products (ETB).\n"
    report += "- **Lending Score**: Weighted sum (40% norm. views, 30% norm. posts, 30% norm. price), scaled to 0–100.\n"
    
    report += "\n## Recommendations\n"
    report += "- **High Scores (>70)**: Prioritize for loans due to strong engagement and activity.\n"
    report += "- **Low Scores (<50)**: Investigate low activity or engagement before lending.\n"
    report += "- **Data Gaps**: Collect more posts for vendors with few posts (e.g., @zemencallcenter).\n"
    report += "- **NER Accuracy**: Improve model (F1 ~0.35) to ensure accurate price extraction.\n"
    
    with open("vendor_scorecard_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Report saved to vendor_scorecard_report.md")

# Main execution
if __name__ == "__main__":
    logger.info("Starting vendor analytics engine...")
    posts = load_conll_data("cleaned_labeled_data.conll")
    posts = assign_vendors(posts)
    metadata = assign_metadata([post["post_id"] for post in posts])
    metrics_df = calculate_vendor_metrics(posts, metadata)
    metrics_df = calculate_lending_score(metrics_df)
    logger.info("Vendor Metrics:\n" + metrics_df.to_string())
    generate_report(metrics_df)
    logger.info("Vendor analytics completed.")