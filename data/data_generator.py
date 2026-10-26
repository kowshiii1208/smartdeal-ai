"""
Demonstration Data Generator for SmartDeal AI.
Generates realistic e-commerce products and time-series historical price observations
with fixed random seed for 100% reproducibility.
All records are explicitly labeled with `is_synthetic = 1`.
"""

import random
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd

from config import (
    RANDOM_SEED,
    SAMPLE_PRODUCTS_PATH,
    SAMPLE_HISTORY_PATH,
    PRODUCT_CATEGORIES,
    POPULAR_BRANDS,
    RETAILERS,
)

# Product Templates for High Realism
PRODUCT_TEMPLATES = [
    # Laptops
    ("Apple", "Laptops", "MacBook Air M2 (13.6-inch, 8GB RAM, 256GB SSD, Space Grey)", 114900, 0.12),
    ("Apple", "Laptops", "MacBook Pro M3 (14.2-inch, 16GB RAM, 512GB SSD, Space Black)", 199900, 0.08),
    ("Dell", "Laptops", "Inspiron 15 3530 (Intel Core i5-1335U, 16GB RAM, 512GB SSD)", 68500, 0.22),
    ("Dell", "Laptops", "XPS 13 9315 (Intel Core i7-1250U, 16GB RAM, 512GB SSD, FHD+)", 124990, 0.15),
    ("HP", "Laptops", "Pavilion 15 (AMD Ryzen 7 7730U, 16GB RAM, 512GB SSD, Backlit KB)", 72000, 0.18),
    ("HP", "Laptops", "Victus Gaming Laptop (Intel Core i5-12450H, RTX 3050, 144Hz)", 79999, 0.24),
    ("ASUS", "Laptops", "ROG Zephyrus G14 (AMD Ryzen 9, RTX 4060, 16GB RAM, 1TB SSD)", 159990, 0.14),
    ("ASUS", "Laptops", "Vivobook 15 (Intel Core i3-1215U, 8GB RAM, 512GB SSD)", 49990, 0.28),
    ("Lenovo", "Laptops", "IdeaPad Slim 3 (Intel Core i5-12450H, 16GB RAM, 512GB SSD)", 64990, 0.26),
    ("Lenovo", "Laptops", "LOQ 15 Gaming (AMD Ryzen 7 7840HS, RTX 4050, 16GB, 512GB)", 94990, 0.18),
    ("Acer", "Laptops", "Nitro V 15 (Intel Core i5-13420H, RTX 4050 6GB, 16GB, 512GB)", 82999, 0.20),
    ("Acer", "Laptops", "Aspire Lite (AMD Ryzen 5 5500U, 16GB RAM, 512GB SSD)", 52990, 0.32),

    # Smartphones
    ("Apple", "Smartphones", "iPhone 15 (128GB, Blue)", 79900, 0.14),
    ("Apple", "Smartphones", "iPhone 15 Pro (256GB, Natural Titanium)", 134900, 0.07),
    ("Samsung", "Smartphones", "Galaxy S24 5G (8GB RAM, 128GB, Onyx Black)", 79999, 0.12),
    ("Samsung", "Smartphones", "Galaxy S24 Ultra (12GB RAM, 256GB, Titanium Gray)", 129999, 0.09),
    ("Samsung", "Smartphones", "Galaxy A55 5G (8GB RAM, 128GB, Awesome Iceblue)", 42999, 0.16),
    ("OnePlus", "Smartphones", "OnePlus 12 (16GB RAM, 512GB, Silky Black)", 69999, 0.09),
    ("OnePlus", "Smartphones", "OnePlus 12R (8GB RAM, 128GB, Cool Blue)", 39999, 0.10),
    ("OnePlus", "Smartphones", "OnePlus Nord CE4 5G (8GB RAM, 128GB, Celadon Marble)", 24999, 0.12),
    ("Google", "Smartphones", "Pixel 8 (128GB, Hazel)", 75999, 0.18),
    ("Google", "Smartphones", "Pixel 8a (128GB, Bay Blue)", 52999, 0.15),
    ("Xiaomi", "Smartphones", "Redmi Note 13 Pro+ 5G (12GB RAM, 256GB, Fusion Purple)", 33999, 0.18),
    ("Realme", "Smartphones", "Realme 12 Pro+ 5G (8GB RAM, 256GB, Submarine Blue)", 31999, 0.19),

    # Audio & Headphones
    ("Sony", "Audio & Headphones", "WH-1000XM5 Wireless Active Noise Cancelling Headphones", 34990, 0.20),
    ("Sony", "Audio & Headphones", "WF-1000XM5 True Wireless Noise Cancelling Earbuds", 29990, 0.23),
    ("Sony", "Audio & Headphones", "WH-CH720N Noise Cancelling Wireless Over-Ear Headphones", 14990, 0.33),
    ("Bose", "Audio & Headphones", "QuietComfort 45 Bluetooth Wireless Noise Cancelling Headphones", 32900, 0.18),
    ("Bose", "Audio & Headphones", "QuietComfort Ultra Wireless Earbuds with Spatial Audio", 29900, 0.14),
    ("Sennheiser", "Audio & Headphones", "Momentum 4 Wireless Audiophile Headphones (60h Battery)", 34990, 0.28),
    ("Sennheiser", "Audio & Headphones", "HD 450SE Bluetooth Wireless Active Noise Cancelling", 14990, 0.40),
    ("JBL", "Audio & Headphones", "Live 660NC Wireless Over-Ear Active Noise Cancelling", 14999, 0.42),
    ("JBL", "Audio & Headphones", "Tune 770NC Adaptive Noise Cancelling Wireless Headphones", 9999, 0.35),
    ("boAt", "Audio & Headphones", "Rockerz 450 Bluetooth On-Ear Headphone (15 Hours Playback)", 3990, 0.62),
    ("boAt", "Audio & Headphones", "Nirvana Ion TWS Earbuds (120 Hours Playback, Dual EQ)", 7990, 0.75),
    ("Noise", "Audio & Headphones", "Noise Two Wireless On-Ear Headphones (50 Hours Playtime)", 4999, 0.68),

    # Smartwatches & Wearables
    ("Apple", "Smartwatches & Wearables", "Watch Series 9 GPS 41mm Midnight Aluminium Case", 41900, 0.12),
    ("Apple", "Smartwatches & Wearables", "Watch Ultra 2 GPS + Cellular 49mm Titanium Case", 89900, 0.06),
    ("Samsung", "Smartwatches & Wearables", "Galaxy Watch 6 Bluetooth 44mm (Graphite)", 33999, 0.35),
    ("Samsung", "Smartwatches & Wearables", "Galaxy Watch 6 Classic 47mm Bluetooth (Black)", 40999, 0.30),
    ("Garmin", "Smartwatches & Wearables", "Forerunner 265 GPS Running Smartwatch (AMOLED Display)", 50490, 0.08),
    ("Garmin", "Smartwatches & Wearables", "Instinct 2 Solar Rugged Outdoor Smartwatch", 46990, 0.15),
    ("Fitbit", "Smartwatches & Wearables", "Charge 6 Advanced Health & Fitness Tracker with Google Apps", 14999, 0.18),
    ("Noise", "Smartwatches & Wearables", "ColorFit Pulse 2 Max 1.85'' Display Bluetooth Calling Smartwatch", 5999, 0.75),
    ("Noise", "Smartwatches & Wearables", "Halo Plus 1.46'' Metallic AMOLED Display Smartwatch", 7999, 0.65),
    ("Amazfit", "Smartwatches & Wearables", "GTR 4 Smart Watch with Dual-Band GPS & 14-Day Battery", 23999, 0.30),
    ("Amazfit", "Smartwatches & Wearables", "Active Smart Watch 1.75'' AMOLED Display with AI Coach", 12999, 0.23),

    # Televisions
    ("LG", "Televisions", "OLED evo C3 Series 55-inch 4K Ultra HD Smart TV (OLED55C3PSA)", 169990, 0.28),
    ("LG", "Televisions", "UR75 Series 43-inch 4K Ultra HD Smart LED TV (43UR7500PSC)", 49990, 0.36),
    ("Samsung", "Televisions", "Crystal iSmart 4K Ultra HD 55-inch Smart LED TV", 68900, 0.38),
    ("Samsung", "Televisions", "The Frame Series 55-inch QLED 4K Smart TV with Matte Display", 134900, 0.35),
    ("Sony", "Televisions", "Bravia 55-inch 4K Ultra HD Smart LED Google TV (KD-55X74L)", 99900, 0.39),
    ("Sony", "Televisions", "Bravia XR 65-inch 4K HDR Full Array LED Google TV (XR-65X90L)", 189900, 0.25),
    ("TCL", "Televisions", "55-inch 4K Ultra HD Smart QLED Google TV (55C645)", 74990, 0.48),
    ("Mi", "Televisions", "Xiaomi X Series 50-inch 4K Ultra HD Smart Google TV", 42999, 0.28),
    ("Vu", "Televisions", "The GloLED Series 55-inch 4K Smart LED Google TV", 55000, 0.31),

    # Tablets
    ("Apple", "Tablets", "iPad 10th Gen 10.9-inch (A14 Bionic, Wi-Fi 64GB, Silver)", 39900, 0.15),
    ("Apple", "Tablets", "iPad Air 11-inch M2 chip (Wi-Fi 128GB, Space Grey)", 59900, 0.07),
    ("Samsung", "Tablets", "Galaxy Tab S9 FE 10.9-inch with S Pen (Wi-Fi 128GB)", 44999, 0.26),
    ("Samsung", "Tablets", "Galaxy Tab S9 Ultra 14.6-inch Dynamic AMOLED 2X (256GB)", 108999, 0.18),
    ("Lenovo", "Tablets", "Tab M11 11-inch WUXGA with Pen (8GB RAM, 128GB)", 28000, 0.40),
    ("Xiaomi", "Tablets", "Pad 6 11-inch 2.8K 144Hz Display (Snapdragon 870, 8GB, 256GB)", 39999, 0.30),
    ("OnePlus", "Tablets", "Pad 11.61-inch 144Hz 7:5 ReadFit Display (12GB, 256GB)", 39999, 0.15),

    # Cameras
    ("Canon", "Cameras", "EOS R50 Mirrorless Camera RF-S18-45mm IS STM Lens Kit", 75995, 0.15),
    ("Canon", "Cameras", "EOS R10 24.2 MP Mirrorless Camera with 18-150mm Lens Kit", 116995, 0.12),
    ("Sony", "Cameras", "Alpha 7 IV Full-frame Hybrid Camera Body (33MP, 4K 60p)", 242490, 0.14),
    ("Sony", "Cameras", "ZV-E10 Mirrorless Vlog Camera with 16-50mm Lens Kit", 69990, 0.16),
    ("Nikon", "Cameras", "Z50 Mirrorless Camera with NIKKOR Z DX 16-50mm Lens Kit", 85995, 0.18),
    ("Fujifilm", "Cameras", "X-T30 II Mirrorless Digital Camera with XC15-45mm Lens Kit", 99999, 0.10),
    ("Panasonic", "Cameras", "Lumix G7 16 MP 4K Mirrorless Camera with 14-42mm Lens", 54990, 0.27),

    # Home Appliances
    ("Dyson", "Home Appliances", "V8 Absolute Cordless Vacuum Cleaner (Powerful Suction)", 43900, 0.25),
    ("Dyson", "Home Appliances", "Purifier Hot+Cool Air Purifier HP07 (HEPA H13 Filter)", 66900, 0.18),
    ("Philips", "Home Appliances", "Airfryer HD9252/90 Digital Air Fryer with Rapid Air Tech", 12995, 0.35),
    ("Philips", "Home Appliances", "Smart Wi-Fi Connected Robot Vacuum Cleaner XU3000", 49995, 0.40),
    ("LG", "Home Appliances", "8.0 Kg 5 Star Inverter Fully-Automatic Front Load Washer", 46990, 0.28),
    ("Samsung", "Home Appliances", "28L Convection Microwave Oven with Tandoor Technology", 17990, 0.32),
    ("Bosch", "Home Appliances", "13 Place Settings Series 4 Dishwasher (Intensive Kadhai)", 54990, 0.22),
    ("IFB", "Home Appliances", "8.5 Kg 5 Star Fully-Automatic Front Loading Machine", 47500, 0.24),
]


def generate_demonstration_dataset(num_products: int = 120, history_days: int = 90) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic e-commerce products and time-series historical price observations.
    Fixed random seed ensures reproducible outputs.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 1. Expand templates to reach num_products
    product_records = []
    prod_id_counter = 1001

    # Base date
    end_date = datetime(2026, 10, 26)
    start_date = end_date - timedelta(days=history_days)

    # First use all templates
    selected_templates = []
    while len(selected_templates) < num_products:
        selected_templates.extend(PRODUCT_TEMPLATES)
    selected_templates = selected_templates[:num_products]

    for brand, category, base_name, mrp, typical_discount in selected_templates:
        prod_id = f"PROD-{prod_id_counter}"
        prod_id_counter += 1

        # Real market variance
        price_variation = np.random.uniform(-0.04, 0.04)
        fair_discount = max(0.04, min(0.70, typical_discount + price_variation))
        
        # Fair market reference price (the equilibrium baseline)
        fair_ref_price = round(mrp * (1.0 - fair_discount), -1)

        # Current listed price fluctuates around the fair reference price
        # Some are exceptional deals (-8% to -15%), some are normal (+/-3%), some are above reference (+5% to +12%)
        deal_scenario = np.random.choice(["great_deal", "good_deal", "fair_price", "above_ref"], p=[0.20, 0.35, 0.30, 0.15])
        if deal_scenario == "great_deal":
            current_discount_vs_ref = np.random.uniform(0.08, 0.18)
            current_price = round(fair_ref_price * (1.0 - current_discount_vs_ref), -1)
        elif deal_scenario == "good_deal":
            current_discount_vs_ref = np.random.uniform(0.03, 0.08)
            current_price = round(fair_ref_price * (1.0 - current_discount_vs_ref), -1)
        elif deal_scenario == "fair_price":
            fluct = np.random.uniform(-0.02, 0.02)
            current_price = round(fair_ref_price * (1.0 + fluct), -1)
        else: # above_ref
            premium = np.random.uniform(0.04, 0.14)
            current_price = min(mrp, round(fair_ref_price * (1.0 + premium), -1))

        # Overall discount from MRP
        actual_discount_pct = round(((mrp - current_price) / mrp) * 100, 1)

        rating = round(float(np.random.normal(4.3, 0.35)), 1)
        rating = max(3.2, min(4.9, rating))

        review_count = int(np.random.lognormal(mean=7.5, sigma=1.2))
        review_count = max(45, min(85000, review_count))

        retailer = np.random.choice(RETAILERS, p=[0.40, 0.30, 0.12, 0.10, 0.08])
        timestamp = (end_date - timedelta(hours=random.randint(1, 48))).isoformat()

        product_records.append({
            "product_id": prod_id,
            "product_name": base_name,
            "category": category,
            "brand": brand,
            "original_price": float(mrp),
            "current_price": float(current_price),
            "discount_percentage": float(actual_discount_pct),
            "rating": float(rating),
            "review_count": int(review_count),
            "retailer": retailer,
            "timestamp": timestamp,
            "fair_reference_price": float(fair_ref_price),
            "is_synthetic": 1,
        })

    products_df = pd.DataFrame(product_records)

    # 2. Generate Historical Price Observations
    history_records = []
    
    # We generate observations every 2-3 days over history_days
    observation_dates = []
    curr = start_date
    while curr <= end_date:
        observation_dates.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=random.choice([2, 3]))

    # Ensure the last observation is end_date
    if end_date.strftime("%Y-%m-%d") not in observation_dates:
        observation_dates.append(end_date.strftime("%Y-%m-%d"))

    for _, prod in products_df.iterrows():
        p_id = prod["product_id"]
        ref = prod["fair_reference_price"]
        retailer = prod["retailer"]

        # Product-specific historical drift and volatility
        volatility = np.random.uniform(0.015, 0.04)
        
        # Major sale event window (e.g. days 30 to 40)
        sale_window_start = start_date + timedelta(days=25)
        sale_window_end = start_date + timedelta(days=35)

        for date_str in observation_dates:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Base price is around reference price
            base_noise = np.random.normal(0, volatility)
            sim_price = ref * (1.0 + base_noise)

            # Check if within festive sale event
            if sale_window_start <= dt <= sale_window_end:
                sim_price *= np.random.uniform(0.85, 0.92)  # 8-15% sale dip

            # If date is end_date, lock to the product's current_price
            if date_str == end_date.strftime("%Y-%m-%d"):
                sim_price = prod["current_price"]

            sim_price = round(max(prod["original_price"] * 0.25, sim_price), -1)

            history_records.append({
                "product_id": p_id,
                "observed_date": date_str,
                "price": float(sim_price),
                "retailer": retailer,
                "in_stock": 1 if np.random.rand() > 0.04 else 0,
                "is_synthetic": 1,
            })

    history_df = pd.DataFrame(history_records)

    return products_df, history_df


def save_demonstration_data(force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate and write sample CSVs to data/ directory if not present or forced."""
    if not force and SAMPLE_PRODUCTS_PATH.exists() and SAMPLE_HISTORY_PATH.exists():
        products_df = pd.read_csv(SAMPLE_PRODUCTS_PATH)
        history_df = pd.read_csv(SAMPLE_HISTORY_PATH)
        return products_df, history_df

    products_df, history_df = generate_demonstration_dataset(num_products=120, history_days=90)
    
    products_df.to_csv(SAMPLE_PRODUCTS_PATH, index=False)
    history_df.to_csv(SAMPLE_HISTORY_PATH, index=False)
    
    return products_df, history_df


if __name__ == "__main__":
    p_df, h_df = save_demonstration_data(force=True)
    print(f"Generated {len(p_df)} products and {len(h_df)} price history observations.")
