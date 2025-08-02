import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import faiss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from datetime import datetime
from collections import Counter
import re
import warnings
import gdown
import kagglehub
from kagglehub import KaggleDatasetAdapter
import nltk
import requests
import os

warnings.filterwarnings('ignore')

# ============================================
# ✅ DATA AUGMENTATION MODULE
# ============================================

# Download required NLTK data
try:
    from nltk.corpus import wordnet
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    WORDNET_AVAILABLE = True
except:
    print("⚠️ NLTK WordNet not available, synonym replacement disabled")
    WORDNET_AVAILABLE = False

# Hard Ham Phrase Groups for augmentation
financial_phrases = [
    "I got $100 cashback yesterday", "The bank refunded me $200 already",
    "I earned $150/day last month from freelancing", "Approved for $500 loan finally",
    "Got quick $300 refund after confirmation", "The store gave me $250 cashback",
    "My account got $100 instantly after confirming", "I received instant $400 transfer today",
    "They sent me exclusive $600 grant, lol", "Netflix actually gave me 3 months free"
]
promotion_phrases = [
    "I bought one and got one free, legit deal", "Flash sale 80% off, I already ordered",
    "Exclusive deal worked for me, saved a lot", "Hot sale 2 hours ago, crazy cheap",
    "New collection free shipping, I tried it", "Best price ever for members",
    "Got special coupon, it worked!", "Reserved early and saved 20%",
    "Only 3 items left when I bought mine", "Order now, it's real not fake"
]
lottery_phrases = [
    "I actually won a $1000 voucher at the mall", "I got a free iPhone from the lucky draw",
    "Claimed my $500 Amazon voucher legit", "Won a prize, just showed my ticket",
    "Spun the wheel at the fair and got gifts", "Lucky draw worked for me today",
    "Redeemed my exclusive prize at the shop", "They reminded me to collect my reward",
    "Gift unlocked at the event, so fun", "Jackpot giveaway, real not scam"
]
scam_alert_phrases = [
    "I got unusual login alert, but it was me", "Reset my password after warning, fine now",
    "Got security update mail, confirmed it's real", "Payment failed once, updated and ok now",
    "Had to verify identity, bank confirmed legit", "Netflix on hold but paid, no issue",
    "Bank asked to confirm transaction, was me", "Warning mail yesterday, false alarm",
    "Confirmed within 24h, all safe", "Suspicious access blocked, just me traveling"
]
call_to_action_phrases = [
    "I clicked to confirm and it worked", "Replied YES, bonus legit",
    "Registered before midnight, no scam", "Tapped link, claimed reward legit",
    "Signed up today, limited seat real", "Confirmed immediately, nothing shady",
    "Acted fast, got discount legit", "Verified email, safe and done",
    "Downloaded app, free points real", "Paid within 12 hours, successful"
]
social_engineering_phrases = [
    "Mom, don't worry I sent you $500 hospital bill already", "Hi mom, phone broke but friend helped",
    "Boss asked me to buy gift cards for office, already did", "John, I transferred $300, check it",
    "Cousin stuck abroad, we sent help", "Friend lent me $200 last week, repaid",
    "Lost wallet but someone returned $150", "Urgent cash request yesterday, sorted now",
    "Helped pay fine, friend returned", "Sister's surgery done, paid $400 legit"
]
obfuscated_phrases = [
    "Clicked h3re to win fr€e gift, real promo", "Got r3fund n0w!!! 100% legit",
    "Fr33 reward worked, tried it", "C@shb@ck real, used today",
    "Won prize real, not spam", "Cl@imed b0nus myself, safe",
    "Gift order legit, no scam", "Refund approved @ bank, no issue",
    "Replied N0W got $$$ legit", "Urg3nt confirm done, real bank"
]
hard_ham_phrase_groups = [
    financial_phrases, promotion_phrases, lottery_phrases,
    scam_alert_phrases, call_to_action_phrases,
    social_engineering_phrases, obfuscated_phrases
]

def generate_hard_ham(ham_texts, n=100):
    if not ham_texts or n <= 0: return []
    hard_ham = []
    for _ in range(n):
        try:
            base = random.choice(ham_texts)
            insert_group = random.choice(hard_ham_phrase_groups)
            insert = random.choice(insert_group)
            if random.random() > 0.5:
                hard_ham.append(f"{base}, btw {insert}.")
            else:
                hard_ham.append(f"{insert}. {base}")
        except Exception as e:
            warnings.warn(f"⚠️ Error generating hard ham: {e}")
            continue
    return hard_ham

def synonym_replacement(text, n=1):
    if not WORDNET_AVAILABLE: return text
    try:
        if isinstance(text, list): text = ' '.join(str(item) for item in text)
        elif not isinstance(text, str): text = str(text)
        if not text or not text.strip(): return text
        words = text.split()
        new_words = words.copy()
        candidates = [w for w in words if wordnet.synsets(w)]
        if not candidates: return text
        random.shuffle(candidates)
        replaced_count = 0
        for random_word in candidates:
            try:
                synonyms = wordnet.synsets(random_word)
                if synonyms:
                    synonym = synonyms[0].lemmas()[0].name().replace('_', ' ')
                    if synonym.lower() != random_word.lower():
                        new_words = [synonym if w == random_word else w for w in new_words]
                        replaced_count += 1
                        if replaced_count >= n: break
            except:
                continue
        return " ".join(new_words)
    except Exception as e:
        warnings.warn(f"⚠️ Synonym replacement error: {e}")
        return str(text) if text else ""

def augment_dataset(messages, labels, aug_ratio=0.2, alpha=0.3):
    augmented_messages, augmented_labels = [], []
    if not isinstance(messages, list): messages = list(messages)
    if not isinstance(labels, list): labels = list(labels)
    clean_messages = [str(msg) if not isinstance(msg, list) else ' '.join(str(item) for item in msg) for msg in messages]
    messages = clean_messages
    ham_count = labels.count('ham')
    spam_count = labels.count('spam')
    print(f"📊 Original dataset: Ham={ham_count}, Spam={spam_count}")

    if ham_count >= spam_count:
        ham_messages = [msg for msg, label in zip(messages, labels) if label == 'ham']
        n_hard_ham = int((ham_count - spam_count) * alpha)
        if n_hard_ham > 0 and ham_messages:
            print(f"🎯 Generating {n_hard_ham} hard ham examples...")
            hard_ham_generated = generate_hard_ham(ham_messages, n=n_hard_ham)
            if hard_ham_generated:
                augmented_messages.extend(hard_ham_generated)
                augmented_labels.extend(['ham'] * len(hard_ham_generated))
                print(f"✅ Generated {len(hard_ham_generated)} hard ham examples")
    
    max_aug_syn = int(len(messages) * aug_ratio)
    print(f"🎯 Attempting to generate ~{max_aug_syn} synonym replacement examples...")
    syn_count = 0
    attempts = 0
    max_attempts = len(messages) * 2
    for msg, label in zip(messages, labels):
        if syn_count >= max_aug_syn or attempts >= max_attempts: break
        attempts += 1
        if random.random() > 0.8:
            try:
                aug_msg = synonym_replacement(msg, n=1)
                if (aug_msg != msg and len(aug_msg.strip()) > 0 and len(aug_msg.split()) >= 2):
                    augmented_messages.append(aug_msg)
                    augmented_labels.append(label)
                    syn_count += 1
            except Exception as e:
                warnings.warn(f"⚠️ Error in synonym replacement: {e}")
                continue
    print(f"✅ Generated {syn_count} synonym replacement examples")
    print(f"✅ Total augmented: {len(augmented_messages)} examples")
    return augmented_messages, augmented_labels

# ==============================
# Class for generating hard examples
# ==============================
class HardExampleGenerator:
    def __init__(self, dataset_path, alpha_spam=0.5, alpha_ham=0.3):
        self.dataset_path = dataset_path
        self.alpha_spam = alpha_spam
        self.alpha_ham = alpha_ham
        self.df = pd.read_csv(dataset_path)
        self.spam_groups = self._init_spam_phrases()
        self.ham_groups = self._init_ham_phrases()

    def _init_spam_phrases(self):
        financial_phrases = [
            "you get $100 back", "they refund $200 instantly", "limited $50 bonus for early registration",
            "earn $150/day remote work", "approved for a $500 credit", "quick $300 refund if you confirm",
            "they give $250 cashback if you check in early", "your account gets $100 instantly after confirmation",
            "instant $400 transfer if you reply YES today", "exclusive $600 grant approved for you"
        ]
        promotion_phrases = [
            "limited time offer ends tonight", "buy one get one free today only", "exclusive deal just for you",
            "hot sale up to 80% off", "flash sale starting in 2 hours", "new collection, free shipping worldwide",
            "best price guaranteed for early birds", "special discount coupon for first 100 buyers",
            "reserve now and get extra 20% off", "only 3 items left, order now!"
        ]
        lottery_phrases = [
            "congratulations! you’ve won a $1000 gift card", "you are selected to receive a free iPhone",
            "claim your $500 Amazon voucher now", "winner! reply to confirm your prize",
            "spin the wheel to win exciting gifts", "lucky draw winner – act fast",
            "redeem your exclusive prize today", "final reminder: unclaimed reward waiting",
            "instant gift unlocked, tap to get", "biggest jackpot giveaway this week"
        ]
        scam_alert_phrases = [
            "your account will be suspended unless verified", "unusual login detected, reset password now",
            "security update required immediately", "urgent: payment failed, update details now",
            "verify your identity to avoid account closure", "your Netflix subscription is on hold, confirm payment",
            "important: unauthorized activity detected", "bank alert: confirm transaction or account locked",
            "last warning: confirm within 24 hours", "emergency: suspicious access blocked, verify"
        ]
        call_to_action_phrases = [
            "click here to confirm", "reply YES to activate bonus", "register before midnight and win",
            "tap now to claim your reward", "sign up today, limited seats", "confirm immediately to proceed",
            "act fast, offer expires soon", "verify email to continue", "download the app and get free points",
            "complete payment within 12 hours"
        ]
        social_engineering_phrases = [
            "hey grandma, i need $500 for hospital bills", "hi mom, send money asap, phone broke",
            "boss asked me to buy 3 gift cards urgently", "john, can you transfer $300 now, emergency",
            "it’s me, your cousin, stuck abroad, need help", "friend, please help me with $200 loan",
            "hi, i lost my wallet, send $150 to this account", "urgent! i can’t talk now, send cash fast",
            "help me pay this fine, will return tomorrow", "sister, please pay $400 for my surgery"
        ]
        obfuscated_phrases = [
            "Cl!ck h3re t0 w1n fr€e iPh0ne", "G€t y0ur r3fund n0w!!!", "L!mited 0ff3r: Fr33 $$$ r3ward",
            "C@shb@ck av@il@ble t0d@y", "W!n b!g pr!ze, act f@st", "Cl@im y0ur 100% b0nus",
            "Fr33 g!ft w!th 0rder", "Up t0 $5000 r3fund @pprov3d", "R3ply N0W t0 r3c3ive $$$",
            "Urg3nt!!! C0nfirm d3tails 1mm3di@tely"
        ]
        return [financial_phrases, promotion_phrases, lottery_phrases,
                scam_alert_phrases, call_to_action_phrases,
                social_engineering_phrases, obfuscated_phrases]

    def _init_ham_phrases(self):
        financial_phrases = [
            "I got $100 cashback yesterday", "The bank refunded me $200 already",
            "I earned $150/day last month from freelancing", "Approved for $500 loan finally",
            "Got quick $300 refund after confirmation", "The store gave me $250 cashback",
            "My account got $100 instantly after confirming", "I received instant $400 transfer today",
            "They sent me exclusive $600 grant, lol", "Netflix actually gave me 3 months free"
        ]
        promotion_phrases = [
            "I bought one and got one free, legit deal", "Flash sale 80% off, I already ordered",
            "Exclusive deal worked for me, saved a lot", "Hot sale 2 hours ago, crazy cheap",
            "New collection free shipping, I tried it", "Best price ever for members",
            "Got special coupon, it worked!", "Reserved early and saved 20%",
            "Only 3 items left when I bought mine", "Order now, it’s real not fake"
        ]
        lottery_phrases = [
            "I actually won a $1000 voucher at the mall", "I got a free iPhone from the lucky draw",
            "Claimed my $500 Amazon voucher legit", "Won a prize, just showed my ticket",
            "Spun the wheel at the fair and got gifts", "Lucky draw worked for me today",
            "Redeemed my exclusive prize at the shop", "They reminded me to collect my reward",
            "Gift unlocked at the event, so fun", "Jackpot giveaway, real not scam"
        ]
        scam_alert_phrases = [
            "I got unusual login alert, but it was me", "Reset my password after warning, fine now",
            "Got security update mail, confirmed it’s real", "Payment failed once, updated and ok now",
            "Had to verify identity, bank confirmed legit", "Netflix on hold but paid, no issue",
            "Bank asked to confirm transaction, was me", "Warning mail yesterday, false alarm",
            "Confirmed within 24h, all safe", "Suspicious access blocked, just me traveling"
        ]
        call_to_action_phrases = [
            "I clicked to confirm and it worked", "Replied YES, bonus legit",
            "Registered before midnight, no scam", "Tapped link, claimed reward legit",
            "Signed up today, limited seat real", "Confirmed immediately, nothing shady",
            "Acted fast, got discount legit", "Verified email, safe and done",
            "Downloaded app, free points real", "Paid within 12 hours, successful"
        ]
        social_engineering_phrases = [
            "Mom, don’t worry I sent you $500 hospital bill already", "Hi mom, phone broke but friend helped",
            "Boss asked me to buy gift cards for office, already did", "John, I transferred $300, check it",
            "Cousin stuck abroad, we sent help", "Friend lent me $200 last week, repaid",
            "Lost wallet but someone returned $150", "Urgent cash request yesterday, sorted now",
            "Helped pay fine, friend returned", "Sister’s surgery done, paid $400 legit"
        ]
        obfuscated_phrases = [
            "Clicked h3re to win fr€e gift, real promo", "Got r3fund n0w!!! 100% legit",
            "Fr33 reward worked, tried it", "C@shb@ck real, used today",
            "Won prize real, not spam", "Cl@imed b0nus myself, safe",
            "Gift order legit, no scam", "Refund approved @ bank, no issue",
            "Replied N0W got $$$ legit", "Urg3nt confirm done, real bank"
        ]
        hard_ham_phrase_groups = [
            financial_phrases, promotion_phrases, lottery_phrases,
            scam_alert_phrases, call_to_action_phrases,
            social_engineering_phrases, obfuscated_phrases
        ]
        return hard_ham_phrase_groups
    
    def _generate_sentences(self, base_texts, phrase_groups, n):
        results = []
        for _ in range(n):
            if not base_texts or not phrase_groups: break
            base = random.choice(base_texts)
            insert = random.choice(random.choice(phrase_groups))
            sentence = f"{insert}. {base}" if random.random() < 0.5 else f"{base}, btw {insert}."
            results.append(sentence)
        return results

    def generate_hard_spam(self, output_path):
        num_ham = self.df[self.df["Category"] == "ham"].shape[0]
        num_spam = self.df[self.df["Category"] == "spam"].shape[0]
        if num_spam >= num_ham:
            print("✅ Spam đã đủ, không sinh thêm.")
            return []
        n_generate = int((num_ham - num_spam) * self.alpha_spam)
        if n_generate > 0:
            base_texts = self.df[self.df["Category"] == "ham"]["Message"].sample(n=n_generate, random_state=42).tolist()
            generated = self._generate_sentences(base_texts, self.spam_groups, n_generate)
            pd.DataFrame({"Category": ["spam"] * len(generated), "Message": generated}).to_csv(output_path, index=False)
            print(f"✅ Sinh {len(generated)} hard spam -> {output_path}")
            return generated
        return []

    def generate_hard_ham(self, output_path):
        num_ham = self.df[self.df["Category"] == "ham"].shape[0]
        num_spam = self.df[self.df["Category"] == "spam"].shape[0]
        if num_ham >= num_spam:
            n_generate = int((num_ham - num_spam) * self.alpha_ham)
            if n_generate > 0:
                base_texts = self.df[self.df["Category"] == "ham"]["Message"].sample(n=n_generate, random_state=42).tolist()
                generated = self._generate_sentences(base_texts, self.ham_groups, n_generate)
                pd.DataFrame({"Category": ["ham"] * len(generated), "Message": generated}).to_csv(output_path, index=False)
                print(f"✅ Sinh {len(generated)} hard ham -> {output_path}")
                return generated
        print("✅ Ham đã đủ, không cần sinh thêm.")
        return []

    def generate_synonym_replacement(self, messages, labels, aug_ratio=0.2):
        MAX_AUG = int(len(messages) * aug_ratio)
        augmented_messages, augmented_labels = [], []
        print(f"✅ Synonym Replacement: sinh tối đa {MAX_AUG} câu.")
        for msg, label in zip(messages, labels):
            if len(augmented_messages) >= MAX_AUG: break
            if random.random() > 0.8:
                aug_msg = self.synonym_replacement(msg)
                if aug_msg != msg:
                    augmented_messages.append(aug_msg)
                    augmented_labels.append(label)
        print(f"✅ Đã sinh {len(augmented_messages)} câu augmented thực tế.")
        return augmented_messages, augmented_labels

def load_data_from_kaggle():
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "victorhoward2/vietnamese-spam-post-in-social-network",
        "vi_dataset.csv"
    )
    print(f"Successfully loaded Kaggle dataset with {len(df)} records")
    return df

def load_data_from_gdrive(file_id="1N7rk-kfnDFIGMeX0ROVTjKh71gcgx-7R"):
    output_path = f"gdrive_dataset_{file_id}.csv"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
    df_base = pd.read_csv(output_path)
    
    hard_spam_out = "hard_spam_generated_auto.csv"
    hard_ham_out = "hard_ham_generated_auto.csv"
    gen = HardExampleGenerator(output_path, alpha_spam=1.0, alpha_ham=0.2)
    gen.generate_hard_spam(hard_spam_out)
    gen.generate_hard_ham(hard_ham_out)

    messages = df_base['Message'].tolist()
    labels = df_base['Category'].tolist()
    augmented_msgs, augmented_lbls = gen.generate_synonym_replacement(messages, labels, aug_ratio=0.2)

    df_hard_spam = pd.read_csv(hard_spam_out)
    df_hard_ham = pd.read_csv(hard_ham_out)

    df_synonym = pd.DataFrame({"Category": augmented_lbls, "Message": augmented_msgs})
    df_augmented = pd.concat([df_base, df_hard_spam, df_hard_ham, df_synonym], ignore_index=True)
    print(f"✅ Tổng số mẫu sau augmentation: {df_augmented.shape[0]}")
    
    return df_augmented

def preprocess_dataframe(df):
    print("Preprocessing dataframe...")
    text_column = None
    label_column = None
    text_candidates = ['message', 'text', 'content', 'email', 'post', 'comment', "texts_vi"]
    for col in df.columns:
        if col.lower() in text_candidates or 'text' in col.lower() or 'message' in col.lower():
            text_column = col
            break
    if text_column is None: text_column = df.columns[0]

    label_candidates = ['label', 'class', 'category', 'type']
    for col in df.columns:
        if col.lower() in label_candidates or 'label' in col.lower():
            label_column = col
            break
    if label_column is None: label_column = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    df[text_column] = df[text_column].astype(str).fillna('')
    df = df[df[text_column].str.strip() != '']

    df[label_column] = df[label_column].astype(str).str.lower()
    label_mapping = {
        '0': 'ham', '1': 'spam',
        'ham': 'ham', 'spam': 'spam',
        'normal': 'ham', 'spam': 'spam',
        'legitimate': 'ham', 'phishing': 'spam',
        'not_spam': 'ham', 'is_spam': 'spam'
    }
    df[label_column] = df[label_column].map(label_mapping).fillna(df[label_column])
    label_counts = df[label_column].value_counts()
    print(f"Label distribution: {label_counts.to_dict()}")

    messages = df[text_column].tolist()
    labels = df[label_column].tolist()
    return messages, labels

def load_dataset(source='kaggle'):
    if source == 'kaggle':
        df = load_data_from_kaggle()
    elif source == 'gdrive':
        df = load_data_from_gdrive()
    else:
        raise ValueError("Source must be 'kaggle' or 'gdrive'")
    
    if df is None:
        raise Exception(f"Failed to load data from {source}")

    return preprocess_dataframe(df)

model_name = "intfloat/multilingual-e5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_embeddings(texts, model, tokenizer, device, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i+batch_size]
        batch_texts_with_prefix = [f"passage: {text}" for text in batch_texts]
        batch_dict = tokenizer(batch_texts_with_prefix, max_length=512, padding=True, truncation=True, return_tensors="pt")
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = model(**batch_dict)
            batch_embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())
    return np.vstack(embeddings)

def calculate_class_weights(labels):
    label_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(label_counts)

    class_weights = {}
    for label, count in label_counts.items():
        class_weights[label] = total_samples / (num_classes * count)
    return class_weights

def compute_saliency_scores(query_text, model, tokenizer, device, index, train_metadata, k=10):
    tokens = tokenizer.tokenize(query_text)
    if len(tokens) <= 1: return np.array([1.0])

    query_with_prefix = f"query: {query_text}"
    batch_dict = tokenizer([query_with_prefix], max_length=512, padding=True, truncation=True, return_tensors="pt")
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

    with torch.no_grad():
        outputs = model(**batch_dict)
        original_embedding = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        original_embedding = F.normalize(original_embedding, p=2, dim=1)
        original_embedding = original_embedding.cpu().numpy().astype("float32")

    original_scores, original_indices = index.search(original_embedding, k)
    original_spam_score = sum(s for s, idx in zip(original_scores[0], original_indices[0])
                              if train_metadata[idx]["label"] == "spam")

    saliencies = []
    for i, token in enumerate(tokens):
        token_mask = tokens.copy()
        token_mask[i] = tokenizer.pad_token
        masked_text = tokenizer.convert_tokens_to_string(token_mask)
        masked_query = f"query: {masked_text}"
        masked_batch_dict = tokenizer([masked_query], max_length=512, padding=True, truncation=True, return_tensors="pt")
        masked_batch_dict = {k: v.to(device) for k, v in masked_batch_dict.items()}

        with torch.no_grad():
            outputs = model(**masked_batch_dict)
            masked_embedding = average_pool(outputs.last_hidden_state, masked_batch_dict["attention_mask"])
            masked_embedding = F.normalize(masked_embedding, p=2, dim=1)
            masked_embedding = masked_embedding.cpu().numpy().astype("float32")

        masked_scores, masked_indices = index.search(masked_embedding, k)
        masked_spam_score = sum(s for s, idx in zip(masked_scores[0], masked_indices[0])
                                if train_metadata[idx]["label"] == "spam")
        saliency = original_spam_score - masked_spam_score
        saliencies.append(saliency)

    arr = np.array(saliencies)
    if len(arr) > 1:
        arr = (arr - arr.min()) / (np.ptp(arr) + 1e-12)
    else:
        arr = np.array([1.0])
    return arr

def classify_with_weighted_knn(query_text, model, tokenizer, device, index, train_metadata, class_weights, k=10, alpha=0.5, explain=False):
    query_with_prefix = f"query: {query_text}"
    batch_dict = tokenizer([query_with_prefix], max_length=512, padding=True, truncation=True, return_tensors="pt")
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
    
    with torch.no_grad():
        outputs = model(**batch_dict)
        query_embedding = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        query_embedding = query_embedding.cpu().numpy().astype("float32")

    scores, indices = index.search(query_embedding, k)
    
    saliency_scores, tokens = None, None
    if explain:
        saliency_scores = compute_saliency_scores(query_text, model, tokenizer, device, index, train_metadata, k)
        saliency_weight = np.mean(saliency_scores)
        tokens = tokenizer.tokenize(query_text)
    else:
        saliency_weight = 0.5 # Default to a neutral value if not explaining

    vote_scores = {"ham": 0.0, "spam": 0.0}
    neighbor_info = []

    for i in range(k):
        neighbor_idx = indices[0][i]
        similarity = float(scores[0][i])
        neighbor_label = train_metadata[neighbor_idx]["label"]
        neighbor_message = train_metadata[neighbor_idx]["message"]
        
        weight = (1 - alpha) * similarity * class_weights[neighbor_label] + alpha * saliency_weight
        vote_scores[neighbor_label] += weight
        
        neighbor_info.append({
            "score": similarity,
            "weight": weight,
            "label": neighbor_label,
            "message": neighbor_message
        })

    predicted_label = max(vote_scores, key=vote_scores.get)

    result = {
        "prediction": predicted_label,
        "vote_scores": vote_scores,
        "neighbors": neighbor_info,
        "saliency_weight": saliency_weight,
        "alpha": alpha
    }

    if explain:
        result["tokens"] = tokens
        result["saliency_scores"] = saliency_scores

    return result

def optimize_alpha_parameter(test_embeddings, test_labels, test_metadata, index, train_metadata, class_weights, k=10):
    print("Optimizing alpha parameter...")
    alpha_values = np.arange(0.0, 1.1, 0.1)
    best_alpha = 0.0
    best_accuracy = 0.0
    alpha_results = []
    
    for alpha in tqdm(alpha_values, desc="Testing alpha values"):
        correct = 0
        total = len(test_embeddings)
        
        for i in range(total):
            query_text = test_metadata[i]["message"]
            true_label = test_metadata[i]["label"]
            
            result = classify_with_weighted_knn(
                query_text, model, tokenizer, device, index, train_metadata,
                class_weights, k=k, alpha=alpha, explain=False
            )
            
            if result["prediction"] == true_label:
                correct += 1
        
        accuracy = correct / total
        alpha_results.append((alpha, accuracy))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = alpha
        
        print(f"Alpha: {alpha:.1f}, Accuracy: {accuracy:.4f}")

    print(f"\nBest alpha: {best_alpha:.1f} with accuracy: {best_accuracy:.4f}")
    return best_alpha, alpha_results

def classify_spam_subcategory(spam_texts, model, tokenizer, device):
    if not spam_texts: return []
    
    reference_texts = {
        'spam_quangcao': "khuyến mãi giảm giá sale ưu đãi mua ngay giá rẻ miễn phí quà tặng voucher coupon giải thưởng trúng thưởng cơ hội trúng promotional discount sale offer prize win money gift free deal bargain cheap special limited",
        'spam_hethong': "thông báo cảnh báo tài khoản bảo mật xác nhận cập nhật hệ thống đăng nhập mật khẩu bị khóa hết hạn gia hạn notification alert account security confirm update system login password locked expired renewal verify suspended warning"
    }

    reference_embeddings = {}
    for category, ref_text in reference_texts.items():
        ref_emb = get_embeddings([ref_text], model, tokenizer, device)[0]
        reference_embeddings[category] = ref_emb

    spam_embeddings = get_embeddings(spam_texts, model, tokenizer, device)
    
    subcategories = []
    for i, (text, text_embedding) in enumerate(zip(spam_texts, spam_embeddings)):
        bert_scores = {}
        for category, ref_emb in reference_embeddings.items():
            similarity = np.dot(text_embedding, ref_emb) / (np.linalg.norm(text_embedding) * np.linalg.norm(ref_emb))
            bert_scores[category] = similarity

        if max(bert_scores.values()) < 0.3:
            best_category = 'spam_khac'
        else:
            best_category = max(bert_scores, key=bert_scores.get)
        subcategories.append(best_category)
    return subcategories

def run_enhanced_pipeline(messages, labels, test_size=0.2, use_augmentation=True):
    print("=== Enhanced Spam Classification Pipeline ===")
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    if use_augmentation:
        try:
            augmented_messages, augmented_labels = augment_dataset(messages, labels)
            if augmented_messages:
                original_count = len(messages)
                messages = messages + augmented_messages
                labels = labels + augmented_labels
                print(f"📈 Dataset size: {original_count} → {len(messages)} (+{len(augmented_messages)})")
                y = le.fit_transform(labels)
            else:
                print("ℹ️ No augmented data generated")
        except Exception as e:
            print(f"⚠️ Augmentation failed: {e}")
            print("ℹ️ Continuing with original data...")
    else:
        print("ℹ️ Data augmentation disabled")

    print("Generating embeddings...")
    X_embeddings = get_embeddings(messages, model, tokenizer, device)
    
    metadata = [{"index": i, "message": message, "label": label, "label_encoded": y[i]}
                for i, (message, label) in enumerate(zip(messages, labels))]

    X_train_emb, X_test_emb, train_metadata, test_metadata = train_test_split(
        X_embeddings, metadata, test_size=test_size, random_state=42,
        stratify=[m["label"] for m in metadata]
    )

    print("Creating FAISS index...")
    dimension = X_train_emb.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(X_train_emb.astype("float32"))

    train_labels = [m["label"] for m in train_metadata]
    class_weights = calculate_class_weights(train_labels)
    
    test_labels = [m["label"] for m in test_metadata]
    best_alpha, alpha_results = optimize_alpha_parameter(
        X_test_emb, test_labels, test_metadata, index, train_metadata, class_weights
    )

    print(f"✅ Training complete. Best alpha: {best_alpha:.1f}")

    return {
        "index": index,
        "train_metadata": train_metadata,
        "class_weights": class_weights,
        "best_alpha": best_alpha
    }

def enhanced_spam_classifier_pipeline(user_input, index, train_metadata, class_weights, best_alpha, k=5, explain=False):
    print(f'\n***Classifying: "{user_input}"')
    print(f"***Using alpha={best_alpha:.1f}, k={k}")
    
    result = classify_with_weighted_knn(
        user_input, model, tokenizer, device, index, train_metadata,
        class_weights, k=k, alpha=best_alpha, explain=explain
    )

    prediction = result["prediction"]
    vote_scores = result["vote_scores"]
    
    subcategory = None
    if prediction == "spam":
        subcategories = classify_spam_subcategory([user_input], model, tokenizer, device)
        subcategory = subcategories[0] if subcategories else "spam_khac"
    
    final_result = {
        "prediction": prediction,
        "subcategory": subcategory,
        "vote_scores": vote_scores,
        "neighbors": result["neighbors"],
        "saliency_weight": result["saliency_weight"],
        "alpha": best_alpha
    }

    if explain and result.get("tokens") is not None:
        final_result["tokens"] = result["tokens"]
        final_result["saliency_scores"] = result["saliency_scores"]

    return final_result
