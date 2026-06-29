import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


st.set_page_config(layout="wide",
                   page_title="Product Recommendation",
                   page_icon="🤖")
df = pd.read_csv("buysense_dataset.csv")

with st.sidebar:
    select = option_menu("Main Menu",
                         ["Home","Prediction","Dataset","About"],
                         icons=["house","cpu","table","info-circle"],
                         default_index=0,
                         menu_icon="cast")

if select == "Home":

    st.markdown("""
    <div style="
        background: linear-gradient(135deg,#0F172A,#1E40AF);
        padding:40px;
        border-radius:20px;
        color:white;
        text-align:center;
        box-shadow:0px 5px 20px rgba(0,0,0,0.25);
    ">

    <h1 style="margin-bottom:5px;">🛍 BuySense</h1>

    <h3 style="margin-top:0;">
    Intelligent Product Discovery Engine
    </h3>

    <p style="font-size:18px;">
    Discover the most suitable product category using Artificial Intelligence
    and Machine Learning based on your shopping behaviour, purchase history,
    and interests.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    total_records = len(df)
    total_categories = df["recommended_category"].nunique()
    total_devices = df["preferred_device"].nunique()
    avg_purchase = round(df["past_purchases"].mean(), 1)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📄 Total Records", f"{total_records:,}")
    col2.metric("🛍 Categories", total_categories)
    col3.metric("📱 Devices", total_devices)
    col4.metric("🛒 Avg Purchases", avg_purchase)

    st.divider()

    st.subheader("📊 Shopping Insights")

    st.markdown("#### Category Distribution")

    category = df["recommended_category"].value_counts()

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(
            category.index,
            category.values,
            color=["#667eea", "#764ba2", "#f093fb", "#4facfe", "#43e97b", "#fa709a"],
            edgecolor="white",
            linewidth=0.5
        )
        ax.bar_label(bars, color="white", fontsize=11)
        ax.set_xlabel("Recommended Category", fontsize=11, color="white")
        ax.set_ylabel("Count", fontsize=11, color="white")
        ax.set_title("Category Prediction Confidence", fontsize=13, fontweight="bold", color="white")
        ax.tick_params(axis="x", rotation=45, colors="white")
        ax.tick_params(axis="y", colors="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        plt.tight_layout()

        st.pyplot(fig)

    st.markdown("#### Preferred Shopping Device")
    with col2:
        device = df["preferred_device"].value_counts()
        colors = [
            "#10B981",
            "#EF4444",
            "#8B5CF6"
        ]

        fig, ax = plt.subplots(figsize=(3,2))

        ax.pie(
            device.values,
            labels=device.index,
            autopct="%1.1f%%",
            colors=colors,
            textprops={"color": "white"}
        )
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        plt.tight_layout()
        st.pyplot(fig)

    st.divider()

    st.subheader("✨ Why Choose BuySense?")

    c1, c2, c3 = st.columns(3)

    st.info("""AI Powered \n\n Uses Machine Learning algorithms to understand shopping behaviour 
    and recommend the most suitable product category.""")


    st.info("""
        Instant Prediction\n
        Generate intelligent recommendations 
        within seconds using a trained ML model.
        """)


    st.info("""Personalized Recommendations \n\n Personalized Recommendations are generated according 
    to user preferences and shopping interests.""")

    st.divider()

elif select == "Prediction":

    st.markdown("""
    <div style="
        background: linear-gradient(135deg,#0F172A,#1E40AF);
        padding:40px;
        border-radius:20px;
        color:white;
        text-align:center;
        box-shadow:0px 5px 20px rgba(0,0,0,0.25);
    ">

    <h1 style="margin-bottom:8px;">🎯 Product Category Prediction</h1>

    <h3 style="margin-top:0;">
    AI Powered Shopping Recommendation
    </h3>

    <p style="font-size:18px; line-height:1.7;">
    Fill in your shopping preferences below and let our Machine Learning model
    analyze your interests, buying behavior, and lifestyle to predict the
    most suitable product category for you.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    data = df.copy()

    le_age      = LabelEncoder()
    le_city     = LabelEncoder()
    le_device   = LabelEncoder()
    le_discount = LabelEncoder()
    le_time     = LabelEncoder()
    le_target   = LabelEncoder()

    data["age_group"] = le_age.fit_transform(data["age_group"])
    data["city_tier"] = le_city.fit_transform(data["city_tier"])
    data["preferred_device"] = le_device.fit_transform(data["preferred_device"])
    data["discount_sensitivity"] = le_discount.fit_transform(data["discount_sensitivity"])
    data["purchase_time"] = le_time.fit_transform(data["purchase_time"])
    data["recommended_category"] = le_target.fit_transform(data["recommended_category"])

    X = data[[
        "age_group", "city_tier", "browsing_minutes", "pages_viewed",
        "past_purchases", "avg_order_value", "discount_sensitivity",
        "preferred_device", "purchase_time",
        "interest_electronics", "interest_fashion", "interest_home_decor",
        "interest_books", "interest_fitness", "interest_beauty",
    ]]
    y = data["recommended_category"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    med_browsing  = int(df["browsing_minutes"].median())
    med_pages     = int(df["pages_viewed"].median())
    med_purchases = int(df["past_purchases"].median())
    med_order_val = float(df["avg_order_value"].median())
    most_city     = df["city_tier"].mode()[0]
    most_time     = df["purchase_time"].mode()[0]

    acc = accuracy_score(y_test, model.predict(X_test))
    # st.success(f"✅ Model ready — Training Accuracy: **{acc*100:.1f}%**")
    # st.markdown("---")

    st.divider()

    st.markdown("### 👤 About You")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.selectbox(
            "🎂 Your Age Group",
            le_age.classes_
        )

    with col2:
        device = st.selectbox(
            "📱 Device You Shop On",
            le_device.classes_,
        )

    with col3:
        discount = st.selectbox(
            "🏷️ Are You a Discount Shopper?",
            le_discount.classes_,
        )

    st.markdown("---")
    st.markdown("### ❤️ What Do You Love? Rate your interest")

    interest_col1, interest_col2, interest_col3 = st.columns(3)

    with interest_col1:
        electronics = st.slider("💻 Electronics & Gadgets", 0, 10, 5)
        fashion     = st.slider("👗 Fashion & Clothing",    0, 10, 5)

    with interest_col2:
        home_decor = st.slider("🏠 Home Decor & Furniture", 1, 10, 5)
        books      = st.slider("📚 Books & Education",      1, 10, 5)

    with interest_col3:
        fitness = st.slider("🏋️ Fitness & Sports", 1, 10, 5)
        beauty  = st.slider("💄 Beauty & Skincare", 1, 10, 5)

    st.divider()

    if st.button("🚀 Find My Best Category!", use_container_width=True):

        sample = pd.DataFrame({
            "age_group":            [le_age.transform([age])[0]],
            "city_tier":            [le_city.transform([most_city])[0]],
            "browsing_minutes":     [med_browsing],
            "pages_viewed":         [med_pages],
            "past_purchases":       [med_purchases],
            "avg_order_value":      [med_order_val],
            "discount_sensitivity": [le_discount.transform([discount])[0]],
            "preferred_device":     [le_device.transform([device])[0]],
            "purchase_time":        [le_time.transform([most_time])[0]],
            "interest_electronics": [electronics],
            "interest_fashion":     [fashion],
            "interest_home_decor":  [home_decor],
            "interest_books":       [books],
            "interest_fitness":     [fitness],
            "interest_beauty":      [beauty],
        })

        probs  = model.predict_proba(sample)[0]
        result = pd.DataFrame({
            "Category":   le_target.classes_,
            "Confidence": probs
        }).sort_values("Confidence", ascending=False).reset_index(drop=True)

        top = result.iloc[0]

        emoji_map = {
            "Electronics": "💻",
            "Fashion":     "👗",
            "Home Decor":  "🏠",
            "Books":       "📚",
            "Fitness":     "🏋️",
            "Beauty":      "💄",
        }
        top_emoji = emoji_map.get(top["Category"], "🛍️")

        st.divider()

        st.markdown("## 🎯 Your Recommendation")

        res_col1, res_col2 = st.columns(2)

        # with res_col1:
        st.markdown(
                f"""
                <div style='
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 16px;
                    padding: 32px 24px;
                    text-align: center;
                    color: white;
                '>
                    <div style='font-size: 56px;'>{top_emoji}</div>
                    <div style='font-size: 26px; font-weight: 700; margin-top: 8px;'>{top['Category']}</div>
                    <div style='font-size: 18px; margin-top: 6px; opacity: 0.9;'>
                        {top['Confidence']*100:.1f}% Confident
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # with res_col2:
        st.markdown("#### 📊 All Category Probabilities")

        categories = result["Category"].tolist()
        confidences = (result["Confidence"] * 100).tolist()
        bar_colors = ["#667eea", "#764ba2", "#f093fb", "#4facfe", "#43e97b", "#fa709a"]

        fig, ax = plt.subplots(figsize=(6, 3))

        bars = ax.bar(categories, confidences, color=bar_colors[:len(categories)], edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, color="white")
        ax.set_xlabel("Category", fontsize=11, color="white")
        ax.set_ylabel("Confidence (%)", fontsize=11, color="white")
        ax.set_title("Category Prediction Confidence", fontsize=13, fontweight="bold", color="white")
        ax.set_ylim(0, max(confidences) + 10)
        ax.tick_params(axis="x",rotation=45,colors="white")
        ax.tick_params(axis="y",colors="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")

        st.pyplot(fig)

        st.markdown("#### 🏆 Top 3 Recommendations")
        top3 = result.head(3).copy()
        top3["Confidence"] = top3["Confidence"].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(top3, use_container_width=True, hide_index=True)

        st.markdown("#### 💡 Why this category?")
        top_interest_name = max(
            [("Electronics", electronics), ("Fashion", fashion),
             ("Home Decor", home_decor),   ("Books", books),
             ("Fitness", fitness),         ("Beauty", beauty)],
            key=lambda x: x[1]
        )
        st.info(
            f"Based on your profile, your strongest interest is **{top_interest_name[0]}** "
            f"(rated {top_interest_name[1]}/10). Combined with your age group **{age}** and "
            f"shopping via **{device}**, we recommend **{top['Category']}** for you."
        )

elif select == "Dataset":
    st.markdown("""
    <div style="
        background: linear-gradient(135deg,#0F172A,#1E40AF);
        padding:40px;
        border-radius:20px;
        color:white;
        text-align:center;
        box-shadow:0px 5px 20px rgba(0,0,0,0.25);
    ">

    <h1 style="margin-bottom:8px;">📂 BuySense Dataset</h1>

    <h3 style="margin-top:0;">
    Machine Learning Training Dataset
    </h3>

    <p style="font-size:18px; line-height:1.6;">
    Explore the dataset used to train the BuySense recommendation model.
    This dataset contains user shopping behavior, browsing patterns,
    purchase history, preferences, and the recommended product category
    used for Machine Learning prediction.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.write("")


    st.title("📂 Dataset Overview")
    st.write("Explore the BuySense dataset used for training the Machine Learning model.")

    st.divider()

    total_rows = df.shape[0]
    total_columns = df.shape[1]
    numerical_columns = len(df.select_dtypes(include=["int64", "float64"]).columns)
    categorical_columns = len(df.select_dtypes(include=["object"]).columns)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📄 Total Rows", total_rows)
    col2.metric("📊 Total Columns", total_columns)
    col3.metric("🔢 Numerical", numerical_columns)
    col4.metric("🔤 Categorical", categorical_columns)

    st.divider()

    st.subheader("📋 Dataset Preview")

    st.dataframe(df, use_container_width=True)

    st.divider()

    st.subheader("ℹ Dataset Information")

    info_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(info_df, use_container_width=True)

    st.divider()

    st.subheader("📈 Statistical Summary")

    st.dataframe(df.describe(include="all"), use_container_width=True)

    st.divider()

    st.subheader("🎯 Target Variable Distribution")

    target = df["recommended_category"].value_counts()

    fig, ax = plt.subplots(figsize=(6,4))

    bars = ax.bar(
        target.index,
        target.values,
        color=["#667eea", "#764ba2", "#f093fb", "#4facfe", "#43e97b", "#fa709a"],
        edgecolor="white",
        linewidth=0.5
    )
    ax.bar_label(bars, color="white", fontsize=11)
    ax.set_xlabel("Recommended Category", fontsize=11, color="white")
    ax.set_ylabel("Count", fontsize=11, color="white")
    ax.set_title("Category Prediction Confidence", fontsize=13, fontweight="bold", color="white")
    ax.tick_params(axis="x", rotation=45, colors="white")
    ax.tick_params(axis="y", colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    plt.tight_layout()

    st.pyplot(fig)

    st.info("The Machine Learning model predicts the 'recommended_category' based on user shopping behaviour and preferences.")
elif select == "About":

    st.markdown("""
    <div style="
        background: linear-gradient(135deg,#0F172A,#1E40AF);
        padding:40px;
        border-radius:20px;
        color:white;
        text-align:center;
        box-shadow:0px 5px 20px rgba(0,0,0,0.25);
    ">

    <h1>🛍️ BuySense AI</h1>

    <h3>Smart Shopping Category Recommendation System</h3>

    <p style="font-size:18px;">
    An Intelligent Machine Learning project that predicts the most suitable
    shopping category based on a user's interests and shopping preferences.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    st.header("📌 Project Overview")

    st.write("""
BuySense AI is an intelligent shopping recommendation system developed
using Machine Learning.

The application analyzes a user's shopping interests and predicts
which product category they are most likely to purchase.

Instead of asking users many complicated questions, the system takes only
a few simple inputs and automatically predicts the best shopping category
using a trained Random Forest Machine Learning model.
""")

    st.divider()

    st.header("✨ Key Features")

    col1, col2 = st.columns(2)

    with col1:

        st.success("🎯 Smart Category Prediction")

        st.success("📊 Confidence Score")

        st.success("📈 Probability Chart")

        st.success("🏆 Top 3 Recommendations")

        st.success("⚡ Fast Prediction")

    with col2:

        st.success("🤖 Machine Learning Model")

        st.success("👤 User Friendly Interface")

        st.success("📂 Dataset Analysis")

        st.success("📱 Responsive Design")

        st.success("💡 Shopping Insights")

    st.divider()

    st.header("⚙️ How BuySense AI Works")

    st.write("""
### Step 1️⃣

The user selects:

- Age Group
- Shopping Device
- Discount Preference

### Step 2️⃣

The user rates interest in different shopping categories:

- Electronics
- Fashion
- Home Decor
- Books
- Fitness
- Beauty

### Step 3️⃣

The Machine Learning model processes the information.

### Step 4️⃣

The system predicts:

✅ Best Shopping Category

✅ Confidence Percentage

✅ Top 3 Recommendations

✅ Probability Distribution
""")

    st.divider()

    st.header("🤖 Machine Learning")

    st.info("""
Algorithm Used:

    • Random Forest Classifier

Libraries Used:

    • Pandas

    • Scikit-Learn

    • Matplotlib

    • Streamlit

The Random Forest algorithm learns shopping behaviour from the training
dataset and predicts the most suitable product category for new users.
""")

    st.divider()

    st.header("📋 Input Features")

    feature_df = pd.DataFrame({

        "Feature":[
            "Age Group",
            "Shopping Device",
            "Discount Preference",
            "Electronics Interest",
            "Fashion Interest",
            "Home Decor Interest",
            "Books Interest",
            "Fitness Interest",
            "Beauty Interest"
        ],

        "Purpose":[
            "User demographic",
            "Shopping platform",
            "Buying behaviour",
            "Interest level",
            "Interest level",
            "Interest level",
            "Interest level",
            "Interest level",
            "Interest level"
        ]
    })

    st.dataframe(feature_df,
                 use_container_width=True,
                 hide_index=True)

    st.divider()

    st.header("🛠️ Technologies Used")

    tech1, tech2, tech3 = st.columns(3)

    tech1.metric("Programming", "Python")

    tech2.metric("Framework", "Streamlit")

    tech3.metric("ML Model", "Random Forest")

    st.write("")

    tech4, tech5, tech6 = st.columns(3)

    tech4.metric("Visualization", "Matplotlib")

    tech5.metric("Data", "Pandas")

    tech6.metric("ML Library", "Scikit-Learn")

    st.divider()

    st.header("🎯 Project Objective")

    st.write("""
The objective of BuySense AI is to simplify shopping recommendations
by using Machine Learning.

The system studies user preferences and predicts the most relevant
shopping category, helping users discover products that match
their interests while demonstrating practical applications of
Artificial Intelligence and Machine Learning.
""")

    st.divider()

    st.markdown("""
    <div style="
        background:#111827;
        color:white;
        padding:20px;
        border-radius:12px;
        text-align:center;
    ">

    <h3>🛍️ BuySense AI</h3>

    <p>Smart Shopping Category Recommendation System</p>

    <p>Built with ❤️ using Python, Streamlit & Machine Learning</p>

    </div>
    """, unsafe_allow_html=True)
