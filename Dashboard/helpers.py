# 약물과 그의 기전에 관한 사전 지식 DB
 # desc: MoA별 작용 기전 및 세포 내 영향 요약
 # signal: MoA별 대표적 신호 변화 및 바이오마커
 # assay: MoA별 권장 실험 방법 및 측정 지표

MOA_KNOWLEDGE = {
    "proteasome_inhibitor": {
        "desc": "26S 프로테아좀 활성 억제 → 비정상 단백질 축적 → 세포 스트레스 및 아폽토시스 유도",
        "signal": "UPR(미접힘 단백질 반응), ER 스트레스 활성화, 유비퀴틴화 단백질 축적",
        "assay": "Proteasome-Glo assay, Ubiquitin 축적 Western blot",
    },
    "nfkb_inhibitor": {
        "desc": "NF-kB 전사인자 경로 억제 → 전염증성 유전자 발현 감소",
        "signal": "IkBa 인산화 감소, p65 핵 이동 억제, TNF-a/IL-6 감소",
        "assay": "NF-kB Luciferase reporter assay, ELISA (IL-6, TNF-a)",
    },
    "hdac_inhibitor": {
        "desc": "히스톤 탈아세틸화효소 억제 → 히스톤 아세틸화 증가 → 유전자 발현 광범위 변화",
        "signal": "H3/H4 과아세틸화, 세포주기 정지(G1/G2), p21 발현 증가",
        "assay": "HDAC fluorometric activity assay, Histone acetylation Western blot",
    },
    "egfr_inhibitor": {
        "desc": "EGFR 티로신 키나아제 억제 → 세포 증식 신호(MAPK/PI3K) 차단",
        "signal": "p-EGFR(Tyr1068) 감소, ERK/AKT 인산화 억제",
        "assay": "EGFR kinase activity assay, p-EGFR Western blot",
    },
    "tubulin_inhibitor": {
        "desc": "튜불린 중합/탈중합 방해 → 방추사 형성 억제 → 유사분열 정지",
        "signal": "G2/M 세포주기 정지, 방추사 체크포인트 활성화",
        "assay": "Tubulin polymerization assay, 세포주기 FACS",
    },
    "pi3k_inhibitor": {
        "desc": "PI3K 억제 → AKT/mTOR 하위 신호 차단 → 세포 생존 및 성장 억제",
        "signal": "p-AKT(Ser473) 감소, PI(3,4,5)P3 생성 억제",
        "assay": "PI3K lipid kinase assay, AKT phosphorylation Western blot",
    },
    "mtor_inhibitor": {
        "desc": "mTOR 억제 → 단백질 합성 및 세포 성장 억제 → 자식작용 유도",
        "signal": "p70S6K/4E-BP1 탈인산화, 자식작용 마커(LC3-II) 증가",
        "assay": "mTOR kinase assay, Western blot (p-S6K, 4E-BP1)",
    },
    "akt_inhibitor": {
        "desc": "AKT 세린/트레오닌 키나아제 억제 → 세포 생존 신호 차단 → 아폽토시스(Apoptosis) 증가",
        "signal": "p-AKT(Ser473/Thr308) 감소, FOXO 핵 이동",
        "assay": "AKT kinase assay, Annexin V apoptosis assay",
    },
}


def get_moa_knowledge(moa_name: str) -> dict:
    return MOA_KNOWLEDGE.get(
        moa_name,
        {
            "desc": f"{moa_name.replace('_', ' ').title()} 관련 기전",
            "signal": "해당 경로 관련 신호 탐색 필요",
            "assay": "대상 특이적 assay 설계 권장 (표적 단백질 활성 측정 및 하위 경로 인산화 확인)",
        },
    )


def assign_reliability_grade(auc: float, ap: float, n_pos: int) -> str:
    if auc >= 0.80 and ap >= 0.25 and n_pos >= 20:
        return "High"
    elif auc >= 0.65 and ap >= 0.05 and n_pos >= 8:
        return "Medium"
    return "Low"


def assign_decision_level(prob: float, reliability: str) -> str:
    if reliability == "High" and prob >= 0.25:
        return "우선 검증 권장"
    if reliability == "High" and prob >= 0.10:
        return "우선/보조 검증 후보"
    if reliability == "Medium" and prob >= 0.15:
        return "보조 검증 후보"
    if reliability == "Medium":
        return "보조 검증 후보"
    return "참고 수준"


def get_toxicity_risk(flag_count: int):
    if flag_count == 0:
        return "Low", "초록", "뚜렷한 세포독성 warning signal 없음"
    if flag_count <= 5:
        return "Medium", "노랑", f"{flag_count}개 세포주에서 주의 필요"
    return "High", "빨강", f"{flag_count}개 세포주에서 광범위한 세포독성 가능성"
