#!/usr/bin/env python3
"""
Unified WeChat Health Coach Bot (`wechat_bot.py`).
Supports Render / Cloud dynamic PORT, Dockerfile, and WeChat Official Account Webhooks.
"""

import argparse
import asyncio
from datetime import datetime
import hashlib
import os
import sys
import time
import xml.etree.ElementTree as ET

from fastapi import FastAPI, HTTPException, Request, Response
from google import genai
from google.genai import types
import httpx

app = FastAPI(title="WeChat Comprehensive Health Coach")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
WECHAT_TOKEN = os.environ.get("WECHAT_TOKEN", "health_coach_token")
DOSSIER_PATH = os.environ.get("DOSSIER_PATH", "zhou_health_dossier.md")
MODEL_ID = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

USER_SESSIONS: dict[str, list[types.Content]] = {}
INFLIGHT_TASKS: dict[str, asyncio.Task] = {}
COMPLETED_REPLIES: dict[str, str] = {}


# ============================================================================
# 1. Deterministic Physiological Calculation Tools
# ============================================================================
def calculate_nutrition_and_hydration_targets(
    weight_kg: float = 54.5,
    height_cm: float = 174.0,
    age: int = 20,
    sex: str = "M",
    activity_level: str = "active",
    goal: str = "bulk",
    body_fat_pct: float = 7.0,
) -> dict:
    """Calculate BMI, BMR, TDEE, daily macros (g), and baseline hydration (mL) per ISSN/ACSM."""
    height_m = height_cm / 100.0
    bmi = round(weight_kg / (height_m ** 2), 2)
    lbm_kg = weight_kg * (1.0 - body_fat_pct / 100.0)
    bmr_km = 370.0 + (21.6 * lbm_kg)
    tdee = bmr_km * 1.725
    target_kcal = tdee * 1.10
    protein_g = round(weight_kg * 1.85)
    fat_g = round(weight_kg * 1.05)
    carbs_g = round(max(0.0, target_kcal - protein_g * 4 - fat_g * 9) / 4.0)
    base_water_ml = round(weight_kg * 35.0)

    return {
        "bmi": bmi,
        "bmr_kcal": round(bmr_km),
        "tdee_kcal": round(tdee),
        "target_daily_kcal": round(target_kcal),
        "protein_g": protein_g,
        "fat_g": fat_g,
        "carbohydrate_g": carbs_g,
        "fiber_min_g": round((target_kcal / 1000.0) * 14.0),
        "workout_day_total_water_ml": f"{base_water_ml + 700}-{base_water_ml + 1000} mL",
    }


def append_daily_checkin_to_dossier(
    weight_kg: str,
    sleep_info: str,
    water_info: str,
    diet_info: str,
    workout_info: str,
    readiness_tier: str,
) -> str:
    """Append a daily check-in row to the user's markdown health dossier."""
    today = datetime.now().strftime("%Y-%m-%d")
    row = f"| {today} | {weight_kg} | {sleep_info} | {water_info} | {diet_info} | {workout_info} | {readiness_tier} |\n"
    if os.path.exists(DOSSIER_PATH):
        with open(DOSSIER_PATH, "a", encoding="utf-8") as f:
            f.write(row)
        return f"Logged into {DOSSIER_PATH}."
    return f"Logged: {row}"


# ============================================================================
# 2. Multimodal Gemini Health Coach Analysis
# ============================================================================
def build_system_instruction() -> str:
    dossier = ""
    if os.path.exists(DOSSIER_PATH):
        with open(DOSSIER_PATH, "r", encoding="utf-8") as f:
            dossier = f.read()

    return f"""你是用户的微信专属健康与增肌教练（基于 `comprehensive-health-manager` 技能与 ACSM/ISSN 循证标准）。

【微信对话与判断规则】
1. 语言自然亲切，精简有力（控制在 150-250 字内，清晰分点，避免冗长废话）。
2. 安全与红旗筛查：若出现胸痛、呼吸急促、发热>=38°C、浓茶色尿伴剧烈肌肉酸痛、急性关节刺痛，立即提醒就医。
3. 准备度分级：【绿灯正常练】/【黄灯减量20-30%、补足碳水与水】/【红灯主动休息】。
4. 营养与饮水精算：结合用户增肌基准（20岁男，174cm，54.5kg，日目标 2500-2780 kcal、碳水 405-475g、蛋白 98-110g、水 2600-3000mL），算清本次摄入，直截了当给出【今天还差多少碳水/蛋白质/水】和【下一餐吃什么、今天怎么练】的具体判断。
5. 自动记录：当用户汇报当日汇总时，自动调用 `append_daily_checkin_to_dossier` 写入档案。

【用户当前体测与健康档案】
{dossier}
"""


async def analyze_health_message(user_id: str, user_text: str = "", image_bytes: bytes = None, image_url: str = "") -> str:
    if not GEMINI_API_KEY:
        return "⚠️ 尚未配置 GEMINI_API_KEY 环境变量，请先设置。"

    client = genai.Client(api_key=GEMINI_API_KEY)
    history = USER_SESSIONS.setdefault(user_id, [])

    parts: list[types.Part] = []
    if image_url:
        async with httpx.AsyncClient(timeout=6.0) as http_client:
            img_resp = await http_client.get(image_url)
            img_resp.raise_for_status()
            mime_type = img_resp.headers.get("Content-Type", "image/jpeg")
            parts.append(types.Part.from_bytes(data=img_resp.content, mime_type=mime_type))
        parts.append(types.Part.from_text(text="请识别这张餐食或体测照片，估算热量、碳水、蛋白质、脂肪，并结合我的增肌目标给出判断与建议。"))
    elif image_bytes:
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
        parts.append(types.Part.from_text(text="请识别这张餐食或体测照片，估算热量、碳水、蛋白质、脂肪，并结合我的增肌目标给出判断与建议。"))
    else:
        parts.append(types.Part.from_text(text=user_text))

    history.append(types.Content(role="user", parts=parts))
    if len(history) > 20:
        history[:] = history[-20:]

    response = await asyncio.to_thread(
        client.models.generate_content,
        model=MODEL_ID,
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=build_system_instruction(),
            tools=[calculate_nutrition_and_hydration_targets, append_daily_checkin_to_dossier],
            temperature=0.3,
        ),
    )
    reply_text = response.text or "已记录你的状态。"
    history.append(types.Content(role="model", parts=[types.Part.from_text(text=reply_text)]))
    return reply_text


# ============================================================================
# 3. Webhook Mode (FastAPI)
# ============================================================================
def verify_wechat_signature(signature: str, timestamp: str, nonce: str) -> bool:
    items = sorted([WECHAT_TOKEN, timestamp, nonce])
    sha1 = hashlib.sha1("".join(items).encode("utf-8")).hexdigest()
    return sha1 == signature


@app.get("/wechat")
async def wechat_verify(signature: str = "", timestamp: str = "", nonce: str = "", echostr: str = ""):
    if not verify_wechat_signature(signature, timestamp, nonce):
        raise HTTPException(status_code=403, detail="Invalid signature")
    return Response(content=echostr, media_type="text/plain")


@app.post("/wechat")
async def wechat_message(request: Request, signature: str = "", timestamp: str = "", nonce: str = ""):
    if not verify_wechat_signature(signature, timestamp, nonce):
        raise HTTPException(status_code=403, detail="Invalid signature")

    raw_xml = await request.body()
    root = ET.fromstring(raw_xml.decode("utf-8"))
    msg_type = root.findtext("MsgType", default="")
    from_user = root.findtext("FromUserName", default="")
    to_user = root.findtext("ToUserName", default="")
    msg_id = root.findtext("MsgId", default=f"{from_user}_{timestamp}")

    if msg_id in COMPLETED_REPLIES:
        reply = COMPLETED_REPLIES.pop(msg_id)
    else:
        if msg_id not in INFLIGHT_TASKS:
            if msg_type == "text":
                content = root.findtext("Content", default="").strip()
                INFLIGHT_TASKS[msg_id] = asyncio.create_task(
                    analyze_health_message(from_user, user_text=content)
                )
            elif msg_type == "image":
                pic_url = root.findtext("PicUrl", default="").strip()
                INFLIGHT_TASKS[msg_id] = asyncio.create_task(
                    analyze_health_message(from_user, image_url=pic_url)
                )
            else:
                INFLIGHT_TASKS[msg_id] = asyncio.create_task(
                    asyncio.sleep(0, result="请直接发送文字或餐食/体测照片，我会实时为你分析营养缺口与训练建议！")
                )

        task = INFLIGHT_TASKS[msg_id]
        try:
            reply = await asyncio.wait_for(asyncio.shield(task), timeout=4.5)
            COMPLETED_REPLIES[msg_id] = reply
            INFLIGHT_TASKS.pop(msg_id, None)
        except asyncio.TimeoutError:
            await asyncio.sleep(5.2)
            return Response(content="success", media_type="text/plain")

    now_ts = int(time.time())
    reply_xml = f"""<xml>
<ToUserName><![CDATA[{from_user}]]></ToUserName>
<FromUserName><![CDATA[{to_user}]]></FromUserName>
<CreateTime>{now_ts}</CreateTime>
<MsgType><![CDATA[text]]></MsgType>
<Content><![CDATA[{reply}]]></Content>
</xml>"""
    return Response(content=reply_xml, media_type="application/xml")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)