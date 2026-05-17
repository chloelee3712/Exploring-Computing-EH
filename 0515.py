import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# 데이터 로드 함수
@st.cache_data
def load_data():
    # 절대 경로 직접 입력
    df = pd.read_csv("data/cars.csv")
    return df

# 홈 화면
def cars_home():
    st.title("🚗 자동차 연비 분석 대시보드")

    st.write("""
    이 대시보드는 자동차 성능 데이터를 기반으로  
    **대륙별 평균 연비, 마력(hp)과 연비(mpg) 관계, 차량 무게와 연비 관계, 연도별 평균 연비 변화** 등을 시각화합니다.
    """)

    st.markdown("---")

    st.subheader("📊 주요 기능")
    st.markdown("""
    1. **탐색적 자료분석 (EDA)**  
       - 제조 대륙별 평균 연비 비교
       - 마력과 차량 무게 대비 연비 분포
       - 연도별 연비 변화 추이
       - 그래프를 통해 데이터를 직관적으로 확인할 수 있습니다.

    2. **연비 예측**  
       - 차량의 제원(`hp`, `weightlbs`, `cubicinches`, `cylinders`)을 입력하면  
         머신러닝 모델이 예상 연비를 예측합니다.

    3. **실시간 인터랙티브 시각화**  
       - 슬라이더, 드롭다운 등 Streamlit 위젯을 활용하여  
         조건을 바꾸면 그래프가 **즉시 업데이트**됩니다.
    """)

    st.markdown("---")

    st.subheader("💡 활용 포인트")
    st.info("""
    - 데이터를 시각화하며 **연비에 영향을 주는 변수**를 탐색할 수 있습니다.
    - 사용자가 직접 조건을 조정하면서 **예상 연비 모델의 반응**을 실시간으로 확인할 수 있습니다.
    - Streamlit을 활용해 **웹 기반 데이터 분석 대시보드** 제작을 체험할 수 있습니다.
    """)

    st.markdown("---")

    st.caption("📁 데이터 출처: Kaggle - Auto MPG Dataset")


# 탐색적 분석 화면
def cars_EDA(df):
    st.title("🔍 자동차 연비 분석 (EDA)")

    st.write("""
    이 탭에서는 자동차 성능 데이터를 활용하여  
    **연비(mpg)와 주요 변수들 간의 관계, 대륙별 특성, 연도별 변화** 등을 탐색합니다.
    """)

    # 데이터 미리보기
    st.subheader("📄 데이터 미리보기")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")

    # 1. 대륙별 평균 연비
    st.subheader("🌏 대륙별 평균 연비")
    continent_mpg = df.groupby("continent")["mpg"].mean().reset_index()
    fig1 = px.bar(
        continent_mpg,
        x="continent",
        y="mpg",
        color="mpg",
        title="대륙별 평균 연비",
        color_continuous_scale="Greens",
        labels={"continent": "제조 대륙", "mpg": "평균 연비 (mpg)"}
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.info("""
    💡 **분석 포인트**
    - 대륙별로 차량 특성이 다르며, 미국 차량은 비교적 연비가 낮고 일본 차량은 연비가 높은 편입니다.
    """)

    st.markdown("---")

    # 2. 마력(hp)과 연비(mpg) 관계
    st.subheader("⚡ 마력(hp)과 연비(mpg) 관계")
    fig2 = px.scatter(
        df,
        x="hp",
        y="mpg",
        color="continent",
        size="weightlbs",
        hover_name="continent",
        title="마력 대비 연비 산점도",
        labels={"hp": "마력 (hp)", "mpg": "연비 (mpg)"}
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.info("""
    💡 **분석 포인트**
    - 마력이 높을수록 연비가 낮아지는 경향이 있습니다.
    - 버블 크기는 차량 무게를 나타냅니다.
    """)

    st.markdown("---")

    # 3. 차량 무게와 연비 관계
    st.subheader("⚖️ 차량 무게(weightlbs)와 연비 관계")
    fig3 = px.scatter(
        df,
        x="weightlbs",
        y="mpg",
        color="continent",
        size="hp",
        hover_name="continent",
        title="차량 무게 대비 연비 산점도",
        labels={"weightlbs": "차량 무게 (lbs)", "mpg": "연비 (mpg)"}
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.info("""
    💡 **분석 포인트**
    - 차량 무게가 무거울수록 연비가 낮아지는 경향이 있습니다.
    - 버블 크기는 마력을 나타냅니다.
    """)

    st.markdown("---")

    # 4. 연도별 평균 연비 변화
    st.subheader("📆 연도별 평균 연비 변화")
    year_mpg = df.groupby("year")["mpg"].mean().reset_index()
    fig4 = px.line(
        year_mpg,
        x="year",
        y="mpg",
        title="연도별 평균 연비 추이",
        markers=True,
        labels={"year": "출시 연도", "mpg": "평균 연비 (mpg)"}
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.info("""
    💡 **분석 포인트**
    - 연도별 평균 연비가 점차 개선되는 추세를 확인할 수 있습니다.
    - 자동차 산업의 기술 발전을 반영합니다.
    """)

    st.markdown("---")

    # 5. 연비 등급 분류 (추가 기능) ⭐
    st.subheader("🏆 연비 등급 분류 (추가 기능)")
    
    # 연비 등급 기준 설정
    good_threshold = df['mpg'].quantile(0.67)  # 상위 33%
    poor_threshold = df['mpg'].quantile(0.33)   # 하위 33%
    
    def classify_mpg(mpg):
        if mpg >= good_threshold:
            return "Good 🟢"
        elif mpg >= poor_threshold:
            return "Average 🟡"
        else:
            return "Poor 🔴"
    
    # 등급 분류 추가
    df_classified = df.copy()
    df_classified['Grade'] = df_classified['mpg'].apply(classify_mpg)
    
    # 등급별 차량 수
    grade_counts = df_classified['Grade'].value_counts().reset_index()
    grade_counts.columns = ['Grade', 'Count']
    
    # 파이 차트
    fig5 = px.pie(
        grade_counts,
        values='Count',
        names='Grade',
        title="연비 등급 분포",
        color_discrete_map={
            'Good 🟢': '#2ecc71',
            'Average 🟡': '#f39c12',
            'Poor 🔴': '#e74c3c'
        }
    )
    st.plotly_chart(fig5, use_container_width=True)
    
    # 등급별 통계
    st.write("### 📊 등급별 통계")
    grade_stats = df_classified.groupby('Grade')['mpg'].agg(['count', 'mean', 'min', 'max']).round(2)
    grade_stats.columns = ['차량수', '평균연비', '최소연비', '최대연비']
    st.dataframe(grade_stats, use_container_width=True)
    
    st.info(f"""
    💡 **등급 기준**
    - **Good 🟢**: {good_threshold:.1f} mpg 이상 (상위 33% - 고연비)
    - **Average 🟡**: {poor_threshold:.1f} ~ {good_threshold:.1f} mpg (중간 34% - 중간연비)
    - **Poor 🔴**: {poor_threshold:.1f} mpg 미만 (하위 33% - 저연비)
    """)

    st.markdown("---")

    # 6. 통계 요약
    st.subheader("📊 데이터 통계 요약")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("평균 연비", f"{df['mpg'].mean():.1f} mpg")
    
    with col2:
        st.metric("평균 마력", f"{df['hp'].mean():.0f} hp")
    
    with col3:
        st.metric("평균 무게", f"{df['weightlbs'].mean():.0f} lbs")
    
    with col4:
        st.metric("총 차량 수", len(df))


# 예측 화면
def cars_predict(df):
    st.header("🤖 자동차 연비 예측")
    st.write("선형회귀(Linear Regression) 모델을 활용하여 자동차의 연비(mpg)를 예측합니다.")

    # 입력 변수(X)와 목표 변수(y) 설정
    X = df[["cylinders", "cubicinches", "hp", "weightlbs", "time-to-60"]]
    y = df["mpg"]

    # 학습 데이터와 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 학습
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 예측 성능 표시
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📊 모델 성능 (R²)", f"{score:.3f}")
    with col2:
        st.metric("데이터셋 크기", len(df))

    st.markdown("---")

    # 사용자 입력
    st.subheader("🚗 자동차 정보 입력")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cylinders = st.slider("실린더 수 (cylinders)", 3, 12, 6)
        cubicinches = st.slider("배기량 (cubicinches)", 60, 500, 200)
        hp = st.slider("마력 (horsepower)", 50, 400, 150)
    
    with col2:
        weightlbs = st.slider("무게 (weightlbs)", 1500, 6000, 3000)
        time_to_60 = st.slider("시속 60마일 도달 시간 (초)", 4.0, 20.0, 10.0)

    st.markdown("---")

    # 입력값으로 예측 수행
    input_data = pd.DataFrame({
        "cylinders": [cylinders],
        "cubicinches": [cubicinches],
        "hp": [hp],
        "weightlbs": [weightlbs],
        "time-to-60": [time_to_60]
    })

    mpg_pred = model.predict(input_data)[0]
    
    st.subheader("🎯 예측 결과")
    st.success(f"예상 연비: **{mpg_pred:.2f} mpg** 🚘")
    
    # 입력값 요약
    st.info(f"""
    **입력한 차량 사양:**
    - 실린더: {cylinders}개
    - 배기량: {cubicinches:.0f} cubic inches
    - 마력: {hp} hp
    - 무게: {weightlbs} lbs
    - 0→60mph 시간: {time_to_60:.1f}초
    """)


# 메인 함수
def main():
    st.set_page_config(page_title="자동차 연비 대시보드", layout="wide")
    
    # --- 사이드바 메뉴 ---
    menu = st.sidebar.radio(
        "대시보드 메뉴",
        ["홈", "탐색적 자료분석(EDA)", "연비 예측"]
    )

    # 데이터 로드
    df = load_data()

    # --- 홈 화면 ---
    if menu == "홈":
        cars_home()

    # --- 탐색적 자료분석 화면 ---
    elif menu == "탐색적 자료분석(EDA)":
        cars_EDA(df)

    # --- 연비 예측 화면 ---
    elif menu == "연비 예측":
        cars_predict(df)


if __name__ == "__main__":
    main()
