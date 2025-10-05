import asyncio
import json
from openai import AsyncOpenAI

# Custom API configuration
API_BASE_URL = "http://5.78.122.79:10000/v1"
API_KEY = "sk-sgl-MH7bEVVJlBp3RT_P5cPQ6-KfC1qJElBRCfTDHy40Ue4"
MODEL = "openai/gpt-4.1"  # Using GPT-4.1 with temperature support

# Language mapping
LANGUAGES = {
    "arb.Arab": "Arabic",
    "ces.Latn": "Czech",
    "cmn.Hani": "Chinese (Simplified)",
    "deu.Latn": "German",
    "fra.Latn": "French",
    "ind.Latn": "Indonesian",
    "ita.Latn": "Italian",
    "jpn.Jpan": "Japanese",
    "kor.Hang": "Korean",
    "nld.Latn": "Dutch",
    "pol.Latn": "Polish",
    "por.Latn": "Portuguese",
    "ron.Latn": "Romanian",
    "rus.Cyrl": "Russian",
    "spa.Latn": "Spanish",
    "tur.Latn": "Turkish"
}

async def translate_text(client, text, language_name, template):
    """Translate text using custom API"""
    prompt = template.replace("{language}", language_name).replace("{text_to_translate}", text)

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )

        translation = response.choices[0].message.content.strip()
        return translation
    except Exception as e:
        print(f"Error translating to {language_name}: {e}")
        return None

async def main():
    # Load template
    with open("strongreject_translation_template.txt", "r", encoding="utf-8") as f:
        template = f.read()

    # Load strong_reject.json
    with open("strong_reject.json", "r", encoding="utf-8") as f:
        strong_reject_data = json.load(f)

    # Get English originals
    eng_prompt = strong_reject_data["eng.Latn"]["strongreject_prompt"]
    eng_system = strong_reject_data["eng.Latn"]["strongreject_system"]

    # Initialize client
    client = AsyncOpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

    print(f"Starting parallel translation with model: {MODEL}\n")

    # Create translation tasks for all languages in parallel
    async def translate_language(lang_code, language_name):
        print(f"[{lang_code}] Starting translation to {language_name}...")

        # Translate both prompt and system in parallel
        prompt_task = translate_text(client, eng_prompt, language_name, template)
        system_task = translate_text(client, eng_system, language_name, template)

        translated_prompt, translated_system = await asyncio.gather(prompt_task, system_task)

        if translated_prompt and translated_system:
            print(f"[{lang_code}] [OK] Prompt: {len(translated_prompt)} chars, System: {len(translated_system)} chars")
            return lang_code, translated_prompt, translated_system
        else:
            print(f"[{lang_code}] [FAILED]")
            return lang_code, None, None

    # Run all translations in parallel
    tasks = [translate_language(lang_code, language_name)
             for lang_code, language_name in LANGUAGES.items()]

    results = await asyncio.gather(*tasks)

    # Update data with results
    for lang_code, translated_prompt, translated_system in results:
        if translated_prompt and translated_system:
            strong_reject_data[lang_code]["strongreject_prompt"] = translated_prompt
            strong_reject_data[lang_code]["strongreject_system"] = translated_system
            print(f"[{lang_code}] [COMPLETED]")

    # Save updated JSON
    with open("strong_reject.json", "w", encoding="utf-8") as f:
        json.dump(strong_reject_data, f, ensure_ascii=False, indent=2)

    print("Translation completed. Results saved to strong_reject.json")
    print("\nRunning quality check...")

    # Quick quality check
    for lang_code in LANGUAGES.keys():
        prompt = strong_reject_data[lang_code]["strongreject_prompt"]
        system = strong_reject_data[lang_code]["strongreject_system"]

        issues = []

        # Check placeholders
        if "{forbidden_prompt}" not in prompt or "{response}" not in prompt:
            issues.append("Missing placeholders in prompt")
        if "{forbidden_prompt}" not in system or "{response}" not in system:
            issues.append("Missing placeholders in system")

        # Check for empty
        if not prompt or not system:
            issues.append("Empty translation")

        # Check for unwanted markers
        unwanted = ["Translation:", "Explanation:", "Note:", "Remember:", "Output:"]
        for marker in unwanted:
            if marker in prompt or marker in system:
                issues.append(f"Contains unwanted marker: {marker}")
                break

        if issues:
            print(f"{lang_code}: ISSUES - {', '.join(issues)}")
        else:
            print(f"{lang_code}: [OK]")

if __name__ == "__main__":
    asyncio.run(main())
