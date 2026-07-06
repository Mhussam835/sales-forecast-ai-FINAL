"""
train_model.py

تدريب ومقارنة عدة نماذج لتوقع المبيعات:
Linear Regression, Random Forest, XGBoost, LightGBM.

يستخدم تقسيم Train/Test زمني (Time-based Split) وليس عشوائيًا،
لتفادي تسريب البيانات المستقبلية (Look-ahead Bias).
"""

import logging
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

# استيراد دفاعي: XGBoost و LightGBM قد لا يكونان مثبّتين في كل بيئة
# (يتطلبان اتصال إنترنت للتثبيت). لو غير متاحين، نتخطاهما بتحذير
# بدل أن يتعطل السكربت بالكامل.
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("مكتبة xgboost غير متاحة - سيتم تخطي هذا النموذج")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("مكتبة lightgbm غير متاحة - سيتم تخطي هذا النموذج")


FEATURE_COLUMNS = [
    "Month", "Week", "Quarter", "Is_Holiday_Season",
    "Lag_1", "Lag_4", "Lag_52", "Rolling_Mean_4", "Rolling_Mean_12",
]
TARGET_COLUMN = "Sales"
CATEGORY_COLUMN = "Category"


def prepare_train_test_split(
    df: pd.DataFrame, test_ratio: float = 0.15, date_col: str = "Order Date"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    تقسيم البيانات زمنيًا: كل ما قبل تاريخ القطع للتدريب، والباقي للاختبار.

    هذا يحاكي السيناريو الحقيقي: النموذج يتدرب على الماضي فقط،
    ويُختبر على مستقبل لم "يره" أثناء التدريب إطلاقًا.

    Args:
        df: DataFrame يحتوي على عمود تاريخ مرتب.
        test_ratio: نسبة البيانات الأحدث المخصصة للاختبار (افتراضيًا 15%).
        date_col: اسم عمود التاريخ.

    Returns:
        (train_df, test_df)
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    cutoff_idx = int(len(df) * (1 - test_ratio))
    cutoff_date = df.iloc[cutoff_idx][date_col]

    train_df = df[df[date_col] < cutoff_date]
    test_df = df[df[date_col] >= cutoff_date]

    logger.info(
        "تقسيم زمني عند %s | Train: %d صف | Test: %d صف",
        cutoff_date.date(), len(train_df), len(test_df)
    )
    return train_df, test_df


def encode_categorical(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    تحويل عمود Category إلى One-Hot Encoding.

    نطبّق get_dummies على train و test معًا مبدئيًا لضمان تطابق
    الأعمدة، ثم نعيد فصلهما - يمنع مشكلة شائعة وهي ظهور فئة في
    الاختبار غير موجودة في التدريب (أو العكس).

    Args:
        train_df: بيانات التدريب.
        test_df: بيانات الاختبار.

    Returns:
        (train_encoded, test_encoded, category_dummy_columns)
    """
    combined = pd.concat([train_df, test_df], axis=0, keys=["train", "test"])
    combined = pd.get_dummies(combined, columns=[CATEGORY_COLUMN], prefix="Cat")

    category_dummy_cols = [c for c in combined.columns if c.startswith("Cat_")]

    train_encoded = combined.xs("train")
    test_encoded = combined.xs("test")

    return train_encoded, test_encoded, category_dummy_cols


def get_model_registry() -> dict[str, Any]:
    """
    إرجاع قاموس بالنماذج المتاحة للتدريب.

    يتحقق ديناميكيًا من توفر xgboost و lightgbm ويضيفهما فقط
    إذا كانا مثبّتين.

    Returns:
        قاموس {اسم_النموذج: كائن_النموذج_غير_المدرّب}.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42
        )
    else:
        # بديل توضيحي بنفس فكرة Boosting المتسلسل، متاح دائمًا عبر sklearn
        models["Gradient Boosting (بديل XGBoost)"] = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42
        )

    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1
        )

    return models


def train_all_models(
    train_df: pd.DataFrame, feature_columns: list[str]
) -> dict[str, Any]:
    """
    تدريب كل النماذج المتاحة على بيانات التدريب.

    Args:
        train_df: بيانات التدريب المُرمّزة (بعد encode_categorical).
        feature_columns: أسماء أعمدة الميزات (تشمل Category dummies).

    Returns:
        قاموس {اسم_النموذج: نموذج_مدرّب}.
    """
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]

    models = get_model_registry()
    trained_models = {}

    for name, model in models.items():
        logger.info("تدريب %s ...", name)
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def save_model(model: Any, path: str) -> None:
    """حفظ نموذج مدرّب على القرص باستخدام joblib."""
    joblib.dump(model, path)
    logger.info("تم حفظ النموذج في %s", path)
