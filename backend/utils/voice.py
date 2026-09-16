"""Voice utility for AgroVeyra using gTTS (Google Text-to-Speech)."""

from gtts import gTTS


def text_to_speech(text: str, lang: str = "en", slow: bool = False) -> bytes:
    """Convert text to speech audio bytes.

    Args:
        text: The text string to convert to speech.
        lang: Language code (default: "en").
        slow: Whether to speak slowly (default: False).

    Returns:
        bytes: MP3 audio data.

    Raises:
        Exception: If gTTS fails to generate speech.
    """
    try:
        tts = gTTS(text=text, lang=lang, slow=slow)
        import io
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        print(f"[voice.py] Generated speech for {len(text)} characters")
        return fp.read()
    except Exception as e:
        print(f"[voice.py] Error generating speech: {e}")
        raise


