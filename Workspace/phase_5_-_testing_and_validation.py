# Phase 5 - Testing & Validation
# # 第 5 阶段 - 测试与验证
# This phase focuses on testing and validation, including diagnostics for data leakage, label distributions, and retraining with balanced splits.
# 该阶段侧重于测试和验证，包括数据泄漏、标签分布的诊断以及使用平衡拆分进行重新训练。
# Depends on final_df from phase 1, and other datasets.
# 依赖于第 1 阶段的 final_df 和其他数据集。

# Diagnostics cell: run multiple leakage and training checks (overlap, label dist, preds, chroma, correlations)
# 诊断单元格：运行多个泄漏和训练检查（重叠、标签分布、预测、chroma、相关性）
import hashlib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report

print('Starting diagnostics checks...')


# helper
def hash_series_texts(s):
    return s.dropna().astype(str).map(lambda x: hashlib.md5(x.strip().encode('utf-8')).hexdigest())


results = {}

# 1) Overlap checks using pandas DataFrames if available  # 1）使用可用的 pandas DataFrame 进行重叠检查
try:
    if 'train_df' in globals() and 'val_df' in globals():
        h_train = set(hash_series_texts(train_df['review_text']))
        h_val = set(hash_series_texts(val_df['review_text']))
        txt_overlap = len(h_train & h_val)
        results['train_val_text_overlap'] = txt_overlap
        print('train/val exact text overlap:', txt_overlap)
    else:
        print('train_df or val_df not in globals(); skipping pandas text overlap check')
except Exception as e:
    print('Error computing pandas train/val text overlap:', e)

# If datasets Arrow splits exist (train_split, val_split)  #  如果数据集 Arrow 拆分存在（train_split、val_split）
try:
    if 'train_split' in globals() and 'val_split' in globals():
        def ds_text_hashes(ds, text_col='review_text'):
            seen = set()
            for x in ds[text_col]:
                if x is None:
                    continue
                seen.add(hashlib.md5(str(x).strip().encode('utf-8')).hexdigest())
            return seen


        s1 = ds_text_hashes(train_split)
        s2 = ds_text_hashes(val_split)
        results['ds_train_val_text_overlap'] = len(s1 & s2)
        print('arrow train/val text overlap:', results['ds_train_val_text_overlap'])
    else:
        print('train_split/val_split not present; skipping Arrow overlap check')
except Exception as e:
    print('Error computing Arrow overlap:', e)

# Print existing precomputed overlap variables if present  # 如果存在，则打印现有的预计算重叠变量
for v in ['train_val_overlap', 'train_test_overlap', 'val_test_overlap']:
    if v in globals():
        print(f'{v} (existing variable) =', globals()[v])

# 2) ID overlaps (product_id or any id-like column)  # 2）ID 重叠（product_id 或任何类似 ID 的列）
try:
    if 'train_df' in globals() and 'val_df' in globals():
        id_cols = [c for c in train_df.columns if 'id' in c.lower() or 'product' in c.lower()]
        for c in id_cols:
            a = set(train_df[c].dropna().astype(str))
            b = set(val_df[c].dropna().astype(str))
            ov = len(a & b)
            print(f'ID overlap on {c}:', ov)
            results[f'id_overlap_{c}'] = ov
    else:
        print('train_df/val_df not available for ID overlap checks')
except Exception as e:
    print('Error computing ID overlaps:', e)

# 3) Label distributions  # 3）标签分布
try:
    def print_label_dist_from_df(df, name='df'):
        vc = df['label'].value_counts(dropna=False)
        print(f"{name} label counts:\n", vc.to_dict())
        print(f"{name} proportions:\n", (vc / vc.sum()).round(3).to_dict())


    if 'train_df' in globals():
        print_label_dist_from_df(train_df, 'train')
    if 'val_df' in globals():
        print_label_dist_from_df(val_df, 'val')
    if 'test_df' in globals():
        print_label_dist_from_df(test_df, 'test')
    # datasets  # 数据集
    if 'train_split' in globals():
        try:
            print('train_split label distribution (arrow):', Counter(train_split['label']))
        except Exception:
            pass
    if 'val_split' in globals():
        try:
            print('val_split label distribution (arrow):', Counter(val_split['label']))
        except Exception:
            pass
except Exception as e:
    print('Error printing label distributions:', e)

# 4) Prediction behavior: collapse to majority or balanced?  # 4）预测行为：是否崩溃为多数或平衡？
try:
    y_true = None
    y_pred = None
    # possible sources
    if 'roberta_predictions' in globals() and len(roberta_predictions) > 0:
        y_pred = np.array(roberta_predictions)
    elif 'roberta_confidences' in globals() and len(roberta_confidences) > 0:
        arr = np.array(roberta_confidences)
        if arr.ndim == 2:
            y_pred = arr.argmax(axis=1)
    # true labels  # 真实标签
    if 'val_labels' in globals() and len(val_labels) > 0:
        y_true = np.array(val_labels)
    elif 'val_dataset' in globals():
        try:
            y_true = np.array(val_dataset['label'])
        except Exception:
            pass
    if y_pred is not None and y_true is not None:
        print('pred dist:', np.bincount(y_pred))
        print('true dist:', np.bincount(y_true))
        print('confusion matrix:\n', confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred, digits=4))
        results['prediction_collapse'] = (np.bincount(y_pred).min() == 0)
    else:
        print('predictions or true labels not available for prediction diagnostics')
except Exception as e:
    print('Error computing prediction diagnostics:', e)

# 5) Chroma / retrieval DB sanity: check if collection contains val/test ids  # 5）Chroma / 检索数据库健全性：检查集合是否包含 val/test ids
try:
    if 'product_profile_collection' in globals():
        coll = product_profile_collection
        try:
            # try to access metadatas via get()  # 尝试通过 get() 访问元数据
            info = coll.get()
            metadatas = info.get('metadatas', [])
            coll_ids = set()
            for md in metadatas:
                if isinstance(md, dict) and 'id' in md:
                    coll_ids.add(str(md['id']))
            print('Chroma collection metadata ids found:', len(coll_ids))
            # compare with val ids if exist  # 如果存在，则与 val id 进行比较
            if 'val_df' in globals():
                val_ids = set(val_df['product_id'].dropna().astype(str)) if 'product_id' in val_df.columns else set()
                print('overlap of Chroma collection with val_df product_id:', len(coll_ids & val_ids))
        except Exception as e:
            print('Could not access collection.get() result directly, trying safer paths:', e)
    else:
        print('product_profile_collection not in globals(); skipping Chroma checks')
except Exception as e:
    print('Error during Chroma checks:', e)

# 6) Feature correlation checks in train_df numeric columns  # 6）train_df 数值列中的特征相关性检查
try:
    if 'train_df' in globals():
        num_cols = train_df.select_dtypes(include=['int', 'float']).columns.tolist()
        suspicious = []
        for c in num_cols:
            if c == 'label':
                continue
            corr = train_df[c].corr(train_df['label'])
            if pd.notna(corr) and abs(corr) > 0.6:
                suspicious.append((c, corr))
        if suspicious:
            print('Highly correlated numeric features (possible leakage):', suspicious)
        else:
            print('No numeric features with correlation > 0.6 found in train_df')
    else:
        print('train_df not available for correlation checks')
except Exception as e:
    print('Error computing feature correlations:', e)

# 7) Majority baseline  # 7）多数基线
try:
    if 'val_df' in globals() and 'label' in val_df.columns:
        maj = val_df['label'].mode()[0]
        maj_acc = (val_df['label'] == maj).mean()
        print('majority class in val:', maj, 'majority baseline accuracy:', round(maj_acc, 4))
    elif y_true is not None:
        vals, counts = np.unique(y_true, return_counts=True)
        maj = vals[np.argmax(counts)]
        maj_acc = counts.max() / counts.sum()
        print('majority class baseline (from y_true):', maj, maj_acc)
except Exception as e:
    print('Error computing majority baseline:', e)

# 8) Quick check for tokenization/preprocessing leakage: did tokenization use full dataset?  # 8）检查标记化/预处理泄漏：标记化是否使用了完整数据集？
try:
    # Look for tokenized_datasets or tokenizer fitted on full corpus before splitting
    if 'tokenized_datasets' in globals():
        print('tokenized_datasets keys:', list(tokenized_datasets.keys()))
        # no direct proof of leakage here, just a note  # 这里没有直接的泄漏证据，只是一个注释
        print('Note: if tokenization/feature stats were computed on full corpus before splitting, that can leak. Check code cells where tokenizer or vectorizer is fit.')
    else:
        print('tokenized_datasets not present; cannot check tokenization source programmatically')
except Exception as e:
    print('Error checking tokenization:', e)

# Synthesize a short diagnosis from checks  # 从检查中综合简短诊断
diag = []
if results.get('train_val_text_overlap', 0) > 0 or results.get('ds_train_val_text_overlap', 0) > 0:
    diag.append('DATA LEAKAGE: exact text overlap between train and validation detected.')
if any((results.get(k, 0) > 0) for k in list(results.keys()) if k.startswith('id_overlap_')):
    diag.append('POTENTIAL LEAKAGE: shared ids between train and validation (see id overlap counts).')
# prediction collapse check  # 预测崩溃检查
if results.get('prediction_collapse', False):
    diag.append('MODEL COLLAPSE: predictions collapse to a single class (majority). Likely severe class imbalance or improper loss/labels.')
# class imbalance check  # 类别不平衡检查
try:
    if 'train_df' in globals():
        vc = train_df['label'].value_counts(normalize=True)
        min_frac = vc.min()
        if min_frac < 0.1:
            diag.append('CLASS IMBALANCE: minority class < 10% in training set. Use class weights or sampling.')
except Exception:
    pass

if not diag:
    diag.append('No smoking-gun leakage detected by these automated checks. Next steps: check manual code cells for uses of validation/test in preprocessing or Chroma indexing. Also try class-weighted training and lower LR.')

print('\n=== CONCISE DIAGNOSIS ===')
for d in diag:
    print('-', d)

print('\nDiagnostics completed.')

# Follow-up evidence cell: print sizes, sample overlaps and prediction lengths to pinpoint leakage source  # 后续证据单元格：打印大小、样本重叠和预测长度以确定泄漏来源
import hashlib
from itertools import islice

print('--- Basic sizes and types ---')
names = ['train_df', 'val_df', 'test_df', 'train_split', 'val_split', 'train_split_indices', 'val_split_indices', 'train_texts', 'val_texts', 'train_labels', 'val_labels', 'roberta_predictions', 'roberta_confidences', 'tokenized_datasets']
for n in names:
    if n in globals():
        v = globals()[n]
        try:
            l = len(v)
        except Exception:
            l = type(v)
        print(f"{n}: type={type(v)}, len={l}")
    else:
        print(f"{n}: MISSING")

# If there are explicit index lists/sets, check intersections  # 如果存在显式索引列表/集合，请检查交集
try:
    if 'train_split_indices' in globals() and 'val_split_indices' in globals():
        s1 = set(train_split_indices)
        s2 = set(val_split_indices)
        inter = s1 & s2
        print('train_split_indices & val_split_indices intersection count:', len(inter))
        if len(inter) > 0:
            print('sample overlapping indices (up to 10):', list(islice(inter, 10)))
except Exception as e:
    print('Could not check index intersections:', e)

# If text arrays/lists exist, show sample overlapping texts  # 如果存在文本数组/列表，请显示样本重叠文本
try:
    if 'train_texts' in globals() and 'val_texts' in globals():
        h1 = {hashlib.md5(t.strip().encode('utf-8')).hexdigest(): t for t in train_texts if t}
        h2 = {hashlib.md5(t.strip().encode('utf-8')).hexdigest(): t for t in val_texts if t}
        common = set(h1.keys()) & set(h2.keys())
        print('text-hash overlap count between train_texts and val_texts:', len(common))
        if len(common) > 0:
            print('Examples of overlapping texts (up to 5):')
            for hh in list(common)[:5]:
                print('-', h1[hh][:200].replace('\n', ' '))
except Exception as e:
    print('Could not compute overlapping texts sample:', e)

# Predictions vs validation labels  # 预测与验证标签
try:
    if 'roberta_predictions' in globals():
        print('len(roberta_predictions)=', len(roberta_predictions))
    if 'val_labels' in globals():
        print('len(val_labels)=', len(val_labels))
    if 'roberta_confidences' in globals():
        import numpy as np

        a = np.array(roberta_confidences)
        print('roberta_confidences shape:', a.shape)
except Exception as e:
    print('Error checking predictions lengths:', e)

# Print precomputed overlap variables  # 打印预计算的重叠变量
for v in ['train_val_overlap', 'train_test_overlap', 'val_test_overlap', 'train_test_overlap']:
    if v in globals():
        print(f'{v} =', globals()[v])

print('\nFollow-up check complete. If any overlaps > 0 above, you have data-split leakage and should rebuild splits to ensure uniqueness.')

# Rebuild balanced splits robustly and retrain (safer fallback behavior)  # 重新构建平衡拆分并重新训练（更安全的回退行为）
import random
import re
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
import torch
import torch.nn as nn
from transformers import TrainingArguments, Trainer
from sklearn.metrics import classification_report, confusion_matrix

print('Start: rebuild balanced dataset and retrain (robust)')

# 1) locate source DataFrame  # 1）定位源 DataFrame
df_candidates = ['final_df', 'combined_df', 'merged_df', 'reviews_clean', 'train_df']
src_df = None
for c in df_candidates:
    if c in globals():
        v = globals()[c]
        if isinstance(v, pd.DataFrame) and len(v) > 0:
            src_df = v.copy()
            src_name = c
            break
if src_df is None:
    raise RuntimeError('No suitable source DataFrame found among: ' + ','.join(df_candidates))
print('Using', src_name, 'with', len(src_df), 'rows')

# 2) infer text column and label column, prefer explicit names  # 2）推断文本列和标签列，优先考虑显式名称
text_col = None
label_col = None
if 'review_text' in src_df.columns:
    text_col = 'review_text'
else:
    text_cols = [c for c in src_df.columns if 'review' in c.lower() or 'text' in c.lower() or 'body' in c.lower()]
    text_col = text_cols[0] if text_cols else None

if 'is_fake' in src_df.columns:
    label_col = 'is_fake'
else:
    label_cols = [c for c in src_df.columns if c.lower() in ('label', 'is_fake', 'fake', 'target', 'y') or set(src_df[c].dropna().unique()).issubset({0, 1})]
    label_col = label_cols[0] if label_cols else None

if text_col is None or label_col is None:
    raise RuntimeError(f'Could not infer text or label columns. text_col={text_col}, label_col={label_col}')

print('Inferred text col =', text_col, 'label col =', label_col)

# Normalize labels to 0/1 where possible  # 将标签规范化为 0/1（如果可能）
if src_df[label_col].dtype == object:
    src_df[label_col] = src_df[label_col].astype(str).map(lambda s: 1 if re.search('fake|fraud|synthetic|bot', s, re.I) else 0)

# 3) deduplicate by text and drop exact duplicates to avoid leakage  # 3）按文本去重并删除完全重复项以避免泄漏
import hashlib

src_df['text_hash'] = src_df[text_col].fillna('').astype(str).map(lambda s: hashlib.md5(s.strip().encode('utf-8')).hexdigest())
src_df = src_df.drop_duplicates(subset=['text_hash']).reset_index(drop=True)
print('After dedup by text_hash:', len(src_df))

# 4) ensure there are both classes — if not, synthesize fake examples by augmenting real texts  # 4）确保存在两类——如果没有，则通过增强真实文本来合成假例
counts = src_df[label_col].value_counts()
print('Counts before synthetic balancing:\n', counts)
if len(counts) < 2 or counts.min() == 0:
    # Synthesize fake examples if missing  # 如果缺失则合成假例
    print('Warning: only one class present or minority class missing; synthesizing fake examples from real texts')
    real_df = src_df[src_df[label_col] == 0].copy()
    if real_df.empty:
        raise RuntimeError('No real examples to synthesize from')
    needed = max(1, int(0.1 * len(real_df)))


    def augment_text_simple_local(s):
        if not isinstance(s, str) or len(s.strip()) == 0:
            return s
        sentences = re.split(r'(?<=[.!?]) +', s)
        if len(sentences) > 1 and random.random() < 0.5:
            random.shuffle(sentences)
            return ' '.join(sentences)
        words = s.split()
        if len(words) > 6 and random.random() < 0.5:
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
            return ' '.join(words)
        return s + ' ' + random.choice(['Great product.', 'Would buy again.', 'Works as expected.'])


    synth_texts = [augment_text_simple_local(t) for t in real_df[text_col].sample(n=needed, replace=True, random_state=42).tolist()]
    synth_df = pd.DataFrame({text_col: synth_texts, label_col: [1] * len(synth_texts)})
    src_df = pd.concat([src_df, synth_df], ignore_index=True)
    print('After synthesis, class counts:', src_df[label_col].value_counts().to_dict())

# 5) build pool and split into train/val (80/20) with stratify  # 5）构建池并拆分为训练/验证（80/20）与分层
pool_df = src_df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'}).sample(frac=1, random_state=42).reset_index(drop=True)
print('Pool size:', len(pool_df), 'class counts:', pool_df['label'].value_counts().to_dict())
from sklearn.model_selection import train_test_split

train_pool, val_pool = train_test_split(pool_df, test_size=0.2, stratify=pool_df['label'], random_state=42)
print('Train pool size:', len(train_pool), 'Val pool size:', len(val_pool))

# 6) Build HuggingFace Datasets and tokenize  # 6）构建 HuggingFace 数据集并进行标记化
if 'tokenizer' not in globals():
    raise RuntimeError('No tokenizer found in notebook environment. Please load tokenizer before running this cell.')

hf_train = Dataset.from_pandas(train_pool.rename(columns={'text': 'text', 'label': 'label'}))
hf_val = Dataset.from_pandas(val_pool.rename(columns={'text': 'text', 'label': 'label'}))

hf_train = hf_train.map(lambda ex: tokenizer(ex['text'], truncation=True, padding='max_length', max_length=256), batched=True)
hf_val = hf_val.map(lambda ex: tokenizer(ex['text'], truncation=True, padding='max_length', max_length=256), batched=True)

hf_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
hf_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 7) compute class weights  # 7）计算类权重
y_train = np.array(hf_train['label'])
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print('Class weights:', dict(zip(classes.tolist(), class_weights.tolist())))


# 8) Weighted Trainer (compat accepts extra kwargs from HF Trainer)  # 8）加权训练器（兼容接受来自 HF Trainer 的额外 kwargs）
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get('labels') if 'labels' in inputs else inputs.get('label')
        outputs = model(input_ids=inputs.get('input_ids'), attention_mask=inputs.get('attention_mask'))
        logits = outputs.logits
        if labels is None:
            # fallback to model-provided loss  # 回退到模型提供的损失
            loss = getattr(outputs, 'loss', torch.tensor(0.0, device=next(model.parameters()).device))
        else:
            loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor.to(logits.device))
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# 9) training arguments (compatibility-safe)  # 9）训练参数（兼容性安全）
training_args = TrainingArguments(
    output_dir='./data/roberta_balanced_check',
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=20,
)

from sklearn.metrics import f1_score, precision_score, recall_score


def compute_metrics_small(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average='binary')
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    return {'f1': f1, 'precision': precision, 'recall': recall}


# 10) load or reuse model  # 10）加载或重用模型
if 'model' not in globals():
    if 'model_dir' in globals():
        from transformers import RobertaForSequenceClassification

        model = RobertaForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    else:
        raise RuntimeError('No model or model_dir found in the notebook environment.')

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=hf_train,
    eval_dataset=hf_val,
    compute_metrics=compute_metrics_small,
)

print('Starting short retraining (1 epoch) for balanced check...')
train_result = trainer.train()
print('Training completed. Metrics:')
print(train_result.metrics)

# Evaluate  # 评估
eval_out = trainer.predict(hf_val)
logits = eval_out.predictions
preds = np.argmax(logits, axis=1)
labels = eval_out.label_ids
print('Confusion matrix:\n', confusion_matrix(labels, preds))
print('Classification report:\n', classification_report(labels, preds, digits=4))

# Save small balanced splits  # 保存小型平衡拆分
train_pool.to_csv('./data/balanced_train_pool.csv', index=False)
val_pool.to_csv('./data/balanced_val_pool.csv', index=False)
print('Saved balanced_train_pool.csv and balanced_val_pool.csv')

print('Retrain cell finished (robust path)')

# Quick inspect cell: show columns, sample rows, and sizes of fake/real lists to choose correct text/label columns
# 快速检查单元格：显示列、样本行和假/真实列表的大小，以选择正确的文本/标签列
import pandas as pd
import inspect

candidates = ['combined_df', 'merged_df', 'final_df', 'reviews_clean', 'train_df']
for name in candidates:
    if name in globals():
        df = globals()[name]
        if isinstance(df, pd.DataFrame):
            print('---', name, '---')
            print('shape:', df.shape)
            print('columns:', df.columns.tolist())
            # show up to first 5 rows but only string/object cols and numeric cols that look like labels
            obj_cols = [c for c in df.columns if df[c].dtype == object]
            sample_cols = obj_cols[:5]
            if 'label' in df.columns:
                sample_cols = list(dict.fromkeys(['label'] + sample_cols))
            print('showing first 5 rows for cols:', sample_cols)
            display(df[sample_cols].head(5))
            # show value counts for any likely label cols  # 显示任何可能的标签列的值计数
            possible_label_cols = [c for c in df.columns if c.lower() in ('label', 'is_fake', 'fake', 'target', 'y') or set(df[c].dropna().unique()).issubset({0, 1})]
            for lc in possible_label_cols:
                print('value counts for', lc, '\n', df[lc].value_counts(dropna=False).to_dict())

# inspect in-memory lists if present  # 如果存在，请检查内存列表
for list_name in ['fake_examples', 'real_examples', 'fake_examples_list', 'debug_fake', 'debug_real']:
    if list_name in globals():
        v = globals()[list_name]
        try:
            print(list_name, 'len =', len(v))
            print('sample (up to 3):', v[:3])
        except Exception as e:
            print('Could not print', list_name, e)

# show tokenizer info if available  # 如果可用，显示标记器信息
if 'tokenizer' in globals():
    try:
        print('Tokenizer type:', type(tokenizer))
        if hasattr(tokenizer, 'vocab_size'):
            print('vocab_size:', tokenizer.vocab_size)
    except Exception as e:
        print('Could not inspect tokenizer:', e)

# show a small sample of model and trainer  # 显示模型和训练器的小样本
if 'model' in globals():
    print('Model present:', model.__class__)
if 'trainer' in globals():
    print('Trainer present with args:', getattr(trainer, 'args', None))

print('\nQuick inspect complete. Use these outputs to pick correct text/label columns for rebuilding.')

# Rebuild splits explicitly using `final_df` (review_text, is_fake), create 50/50 training set, 80/20 realistic validation, tokenize, and retrain
# 使用 `final_df`（review_text，is_fake）显式重建拆分，创建 50/50 训练集，80/20 现实验证，标记化并重新训练
import random
import re
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch
import torch.nn as nn
from transformers import TrainingArguments, Trainer

print('Starting explicit rebuild + retrain using final_df')
# 1) load final_df  # 1）加载 final_df
if 'final_df' not in globals():
    raise RuntimeError('`final_df` not found in notebook namespace. It must contain `review_text` and `is_fake`.')
fd = final_df.copy()
print('final_df shape:', fd.shape)
if 'review_text' not in fd.columns or 'is_fake' not in fd.columns:
    raise RuntimeError('final_df must contain columns `review_text` and `is_fake`')

# 2) basic cleaning: drop NaN texts  # 2）基本清理：删除 NaN 文本
fd['review_text'] = fd['review_text'].astype(str).fillna('').map(lambda s: s.strip())
fd = fd[fd['review_text'].str.len() > 0].reset_index(drop=True)
print('after dropping empty texts:', len(fd))

# 3) build validation set of size ~20% of data with 80% real and 20% fake  # 3）构建大小约为数据 20% 的验证集，其中 80% 为真实，20% 为假
total = len(fd)
val_size = int(0.2 * total)
print('target val size:', val_size)
real_pool = fd[fd['is_fake'] == 0].copy()
fake_pool = fd[fd['is_fake'] == 1].copy()
print('real_pool:', len(real_pool), 'fake_pool:', len(fake_pool))

# compute target counts  # 计算目标计数
val_real_target = int(round(val_size * 0.8))
val_fake_target = val_size - val_real_target
val_real = real_pool.sample(n=min(val_real_target, len(real_pool)), random_state=42)
val_fake = fake_pool.sample(n=min(val_fake_target, len(fake_pool)), random_state=42)
# if not enough fake examples for val_fake_target, adjust by taking as many as available  # 如果没有足够的假例用于 val_fake_target，则通过获取可用的尽可能多来进行调整
if len(val_fake) < val_fake_target:
    shortage = val_fake_target - len(val_fake)
    print('Warning: not enough fake examples for desired val composition; shortage:', shortage)
    # reduce val_real to keep val_size constant  # 减少 val_real 以保持 val_size 恒定
    if len(val_real) > shortage:
        val_real = val_real.sample(n=(len(val_real) - shortage), random_state=42)

val_df = pd.concat([val_real, val_fake]).sample(frac=1, random_state=42).reset_index(drop=True)
print('Actual val composition:', val_df['is_fake'].value_counts().to_dict())

# 4) remaining df for training  # 4）剩余 df 用于训练
val_idx = set(val_df.index)  # these are from subsets, but we need to remove by indices in original df  # 这些来自子集，但我们需要通过原始 df 中的索引进行删除
# Remove val samples from fd by text hashes to avoid accidental duplicates  # 通过文本哈希从 fd 中删除 val 样本以避免意外重复
val_hashes = set(val_df['review_text'].map(lambda s: hashlib.md5(s.encode('utf-8')).hexdigest()))
fd['text_hash'] = fd['review_text'].map(lambda s: hashlib.md5(s.encode('utf-8')).hexdigest())
train_pool_df = fd[~fd['text_hash'].isin(val_hashes)].copy().reset_index(drop=True)
print('Train pool after removing val hashes:', len(train_pool_df), 'class counts:', train_pool_df['is_fake'].value_counts().to_dict())

# 5) create balanced training set 50/50 by oversampling minority with light augmentation  # 5）通过轻度增强少数群体创建 50/50 平衡训练集
real_train = train_pool_df[train_pool_df['is_fake'] == 0].copy()
fake_train = train_pool_df[train_pool_df['is_fake'] == 1].copy()
print('Available for train - real:', len(real_train), 'fake:', len(fake_train))


def augment_text_simple(s):
    if not isinstance(s, str) or len(s.strip()) == 0:
        return s
    # shuffle sentences sometimes
    sentences = re.split(r'(?<=[.!?]) +', s)
    if len(sentences) > 1 and random.random() < 0.4:
        random.shuffle(sentences)
        return ' '..join(sentences)
    words = s.split()
    if len(words) > 6 and random.random() < 0.4:
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
        return ' '..join(words)
    return s + ' '


# target per-class = max(len(real_train), len(fake_train))? We want balanced, so choose target = max(len(real_train), len(fake_train))  # 目标类别 = max(len(real_train), len(fake_train))？我们希望类别平衡，所以选择 target = max(len(real_train), len(fake_train))。
# but to avoid huge oversampling, cap target at available majority count  # 但为了避免过度采样过多，将目标限制在可用的多数计数
target = max(len(real_train), len(fake_train))
if target == 0:
    raise RuntimeError('No data available to form training set after removing validation.')

# build balanced sets  # 构建平衡集
if len(real_train) > len(fake_train):
    # oversample fake to match real  # 过采样假以匹配真实
    needed = len(real_train) - len(fake_train)
    if len(fake_train) > 0:
        extra = fake_train.sample(n=needed, replace=True, random_state=42).copy()
        # augment copies to reduce identical duplicates  # 增强副本以减少完全相同的重复项
        extra['review_text'] = extra['review_text'].map(lambda s: augment_text_simple(s))
        fake_balanced = pd.concat([fake_train, extra], ignore_index=True)
        real_balanced = real_train
    else:
        # synthesize fake examples from real_train by simple augment  # 从 real_train 通过简单增强合成假例
        synth_texts = [augment_text_simple(t) for t in real_train['review_text'].sample(n=len(real_train), replace=True, random_state=42).tolist()]
        fake_balanced = pd.DataFrame({'review_text': synth_texts, 'is_fake': 1})
        real_balanced = real_train
elif len(fake_train) > len(real_train):
    needed = len(fake_train) - len(real_train)
    if len(real_train) > 0:
        extra = real_train.sample(n=needed, replace=True, random_state=42).copy()
        extra['review_text'] = extra['review_text'].map(lambda s: augment_text_simple(s))
        real_balanced = pd.concat([real_train, extra], ignore_index=True)
        fake_balanced = fake_train
else:
    real_balanced = real_train
    fake_balanced = fake_train

train_df = pd.concat([real_balanced[['review_text', 'is_fake']], fake_balanced[['review_text', 'is_fake']]], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
print('Final train counts:', train_df['is_fake'].value_counts().to_dict())

# 6) Tokenize using existing tokenizer  # 6）使用现有标记器进行标记化
if 'tokenizer' not in globals():
    raise RuntimeError('tokenizer not found in notebook namespace; please load tokenizer before running this cell')

from datasets import Dataset

hf_train = Dataset.from_pandas(train_df.rename(columns={'review_text': 'text', 'is_fake': 'label'}))
hf_val = Dataset.from_pandas(val_df.rename(columns={'review_text': 'text', 'is_fake': 'label'}))


def tok_batch(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)


hf_train = hf_train.map(lambda ex: tokenizer(ex['text'], truncation=True, padding='max_length', max_length=256), batched=True)
hf_val = hf_val.map(lambda ex: tokenizer(ex['text'], truncation=True, padding='max_length', max_length=256), batched=True)

hf_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
hf_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 7) compute class weights (based on training distribution)  # 7）计算类权重（基于训练分布）
y_train = np.array(hf_train['label'])
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
print('Computed class weights:', dict(zip(classes.tolist(), class_weights.tolist())))


# 8) Weighted Trainer subclass
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('label')
        outputs = model(input_ids=inputs.get('input_ids'), attention_mask=inputs.get('attention_mask'))
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# 9) Prepare model and training args  # 9）准备模型和训练参数
if 'model' not in globals():
    if 'model_dir' in globals():
        from transformers import RobertaForSequenceClassification

        model = RobertaForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    else:
        raise RuntimeError('No model or model_dir present in environment')

training_args = TrainingArguments(
    output_dir='./data/roberta_balanced_finetune',
    evaluation_strategy='epoch',
    save_strategy='no',
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=False,
)

# 10) compute_metrics  # 10）计算指标
from sklearn.metrics import precision_score, recall_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'f1': f1_score(labels, preds, average='binary'),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0)
    }


# 11) instantiate trainer and train  # 11）实例化训练器并训练
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=hf_train,
    eval_dataset=hf_val,
    compute_metrics=compute_metrics
)

print('Begin training...')
train_result = trainer.train()
print('Training finished, metrics:', train_result.metrics)

# 12) evaluate and print detailed report  # 12）评估并打印详细报告
pred_out = trainer.predict(hf_val)
logits = pred_out.predictions
preds = np.argmax(logits, axis=1)
labels = pred_out.label_ids
print('Validation confusion matrix:\n', confusion_matrix(labels, preds))
print('Validation classification report:\n', classification_report(labels, preds, digits=4))

# 13) update kernel variables for later steps  # 13）更新内核变量以供后续步骤使用
final_train_df = train_df
final_val_df = val_df

print('Retrain cell complete')

# Compatibility cell: create TrainingArguments without unsupported kwargs and run training (uses hf_train, hf_val, model, class_weights_tensor from kernel)
# 兼容性单元格：创建不带不支持的 kwargs 的 TrainingArguments 并运行训练（使用来自内核的 hf_train、hf_val、model、class_weights_tensor）
from transformers import TrainingArguments

print('Creating simpler TrainingArguments for compatibility...')
training_args = TrainingArguments(
    output_dir='./data/roberta_balanced_finetune',
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_steps=50,
)

# compute_metrics should exist from previous cell; if not, define it  # compute_metrics 应该存在于前面的单元格中；如果没有，请定义它
try:
    compute_metrics
except NameError:
    from sklearn.metrics import f1_score, precision_score, recall_score


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            'f1': f1_score(labels, preds, average='binary'),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0)
        }

# instantiate trainer and train  # 实例化训练器并训练
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=hf_train,
    eval_dataset=hf_val,
    compute_metrics=compute_metrics
)
print('Starting training (compatibility run)...')
train_result = trainer.train()
print('Training done. Metrics:', train_result.metrics)

# evaluate  # 评估
pred_out = trainer.predict(hf_val)
logits = pred_out.predictions
preds = np.argmax(logits, axis=1)
labels = pred_out.label_ids
print('Validation confusion matrix:\n', confusion_matrix(labels, preds))
print('Validation classification report:\n', classification_report(labels, preds, digits=4))


# Fix: define a Trainer compute_loss compatible with extra kwargs passed by Trainer (accept **kwargs)  # 修复：定义一个与 Trainer 传递的额外 kwargs 兼容的 Trainer compute_loss（接受 **kwargs）
class WeightedTrainerCompat(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get('label')
        outputs = model(input_ids=inputs.get('input_ids'), attention_mask=inputs.get('attention_mask'))
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# instantiate and train using the compatibility trainer  # 使用兼容性训练器实例化和训练
trainer = WeightedTrainerCompat(
    model=model,
    args=training_args,
    train_dataset=hf_train,
    eval_dataset=hf_val,
    compute_metrics=compute_metrics
)
print('Starting compatible training run...')
train_result = trainer.train()
print('Training finished, metrics:', train_result.metrics)

# evaluate  # 评估
pred_out = trainer.predict(hf_val)
logits = pred_out.predictions
preds = np.argmax(logits, axis=1)
labels = pred_out.label_ids
print('Validation confusion matrix:\n', confusion_matrix(labels, preds))
print('Validation classification report:\n', classification_report(labels, preds, digits=4))


# Fix 2: robust compute_loss that accepts either 'labels' or 'label' and handles None  # 修复 2：强大的 compute_loss，接受“labels”或“label”，并处理 None
class WeightedTrainerCompat2(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # get labels under either key
        labels = inputs.get('labels') if 'labels' in inputs else inputs.get('label')
        # move tensors to device if needed (Trainer will have already moved inputs)
        outputs = model(input_ids=inputs.get('input_ids'), attention_mask=inputs.get('attention_mask'))
        logits = outputs.logits
        if labels is None:
            # fallback to model's own loss if labels absent
            return outputs.loss if return_outputs else outputs.loss
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


trainer = WeightedTrainerCompat2(
    model=model,
    args=training_args,
    train_dataset=hf_train,
    eval_dataset=hf_val,
    compute_metrics=compute_metrics
)
print('Starting robust compatible training run...')
train_result = trainer.train()
print('Training finished, metrics:', train_result.metrics)

pred_out = trainer.predict(hf_val)
logits = pred_out.predictions
preds = np.argmax(logits, axis=1)
labels = pred_out.label_ids
print('Validation confusion matrix:\n', confusion_matrix(labels, preds))
print('Validation classification report:\n', classification_report(labels, preds, digits=4))

# This phase creates files: balanced_train_pool.csv and balanced_val_pool.csv in the data folder, and roberta_balanced_finetune model directory.
# 这些阶段会在数据文件夹中创建文件：balanced_train_pool.csv 和 balanced_val_pool.csv，以及 roberta_balanced_finetune 模型目录。
