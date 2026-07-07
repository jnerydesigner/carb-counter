"""
Agente simples que analisa uma foto de um prato de comida
e retorna os ingredientes identificados, em formato JSON.

Requisitos:
    pip install anthropic

Configuração:
    Defina sua chave de API como variável de ambiente:
        export ANTHROPIC_API_KEY="sua-chave-aqui"
"""

import base64
import json
import mimetypes
import random
import time
from pathlib import Path
from typing import Literal

import anthropic
from anthropic.types import Message, MessageParam
from dotenv import load_dotenv

load_dotenv()

MediaType = Literal["image/jpeg", "image/png", "image/webp", "image/gif"]


def criar_mensagem_com_retry(
    client: anthropic.Anthropic,
    max_tentativas: int = 4,
    espera_base: float = 2.0,
    **kwargs,
) -> Message:
    """Chama client.messages.create com retry e backoff exponencial.

    Cobre casos como a API estar temporariamente fora do ar (5xx),
    rate limit (429) ou falha de conexão.
    """
    ultimo_erro: Exception | None = None

    for tentativa in range(max_tentativas):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as erro:
            ultimo_erro = erro
        except anthropic.APIStatusError as erro:
            if erro.status_code >= 500:
                ultimo_erro = erro
            else:
                raise  # erro do cliente (4xx) não deve ser repetido
        except anthropic.APIConnectionError as erro:
            ultimo_erro = erro

        if tentativa < max_tentativas - 1:
            espera = espera_base * (2**tentativa) + random.uniform(0, 1)
            print(
                f"Falha ao chamar a API (tentativa {tentativa + 1}/{max_tentativas}). "
                f"Tentando novamente em {espera:.1f}s..."
            )
            time.sleep(espera)

    assert ultimo_erro is not None
    raise ultimo_erro


def carregar_imagem_base64(caminho_imagem: str) -> tuple[str, MediaType]:
    """Lê a imagem do disco e retorna (base64_data, media_type)."""
    caminho = Path(caminho_imagem)
    media_type, _ = mimetypes.guess_type(caminho)
    if media_type not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
        raise ValueError(f"Tipo de imagem não suportado: {media_type}")

    dados = caminho.read_bytes()
    return base64.standard_b64encode(dados).decode("utf-8"), media_type


def identificar_ingredientes(caminho_imagem: str) -> dict:
    """Envia a imagem para o Claude e pede a lista de ingredientes em JSON."""
    client = anthropic.Anthropic()  # usa ANTHROPIC_API_KEY do ambiente

    imagem_b64, media_type = carregar_imagem_base64(caminho_imagem)

    prompt = (
        "Você é um assistente especializado em identificar ingredientes de pratos "
        "de comida em fotos. Observe a imagem e responda APENAS com um JSON no "
        "seguinte formato, sem nenhum texto antes ou depois:\n\n"
        '{"ingredientes": ["arroz", "feijão", "frango"], '
        '"confianca": "alta|media|baixa", '
        '"observacoes": "algo relevante, se houver"}\n\n'
        "Liste apenas o que você conseguir identificar com razoável certeza visual, "
        "e seja conciso. Se não conseguir identificar nada, retorne um JSON vazio "
        'como {"ingredientes": [], "confianca": "baixa", "observacoes": "não foi possível identificar ingredientes"}'
        'se houver bebidas tente identificar também, só inclua a marca e o modelo, exemplo: "Coca-Cola 350ml", "Heineken 600ml", "Guaraná Antarctica 2L"'
        'se possivel faça uma contagem de carboidratos, proteínas e gorduras, exemplo: "carboidratos": 20g, "proteínas": 10g, "gorduras": 5g"'
        'estime também o quanto a glicemia deve subir com esse prato, em mg/dL, como uma referência aproximada para '
        'a pessoa ter noção de quanto de correção de insulina pode ser necessária. Inclua no JSON como '
        '"glicose_estimada": 40 (apenas o número, sem unidade)'
    )

    mensagens: list[MessageParam] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": imagem_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    resposta = criar_mensagem_com_retry(
        client,
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=mensagens,
    )

    if resposta.stop_reason == "max_tokens":
        return {
            "erro": "Resposta cortada por atingir max_tokens antes de terminar o JSON",
            "bruto": next(
                (b.text for b in resposta.content if b.type == "text"), ""
            ),
        }

    bloco_texto = next((b for b in resposta.content if b.type == "text"), None)
    if bloco_texto is None:
        return {"erro": "Resposta não contém texto", "bruto": str(resposta.content)}

    texto = bloco_texto.text.strip()

    # remove possíveis blocos de código ```json ... ```
    texto = texto.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        return {"erro": "Resposta não veio em JSON válido", "bruto": texto}


if __name__ == "__main__":
    caminho = "comida2.jpeg"  # troque pelo caminho da sua imagem
    resultado = identificar_ingredientes(caminho)
    saida = json.dumps(resultado, ensure_ascii=False, indent=2)
    print(saida)
    Path("relatorio-2.json").write_text(saida, encoding="utf-8")