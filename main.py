import cv2
import mediapipe as mp
import time
import numpy as np
import os

# lista dos slides
IMAGE_FILENAMES = [
    "slides/1.png",
    "slides/2.png",
    "slides/3.png",
    "slides/4.png",
    "slides/5.png",
    "slides/6.png",
    "slides/7.png",
    "slides/8.png",
    "slides/9.png",
    "slides/10.png",
    "slides/11.png"
]
# define as dimensões das janelas dos slides
SLIDE_WINDOW_WIDTH = 1920
SLIDE_WINDOW_HEIGHT = 1080
SLIDE_WINDOW_NAME = "Slide Atual"

# cooldown reconhecimento de um novo gesto
ACTION_COOLDOWN = 1.5

# são os pontos de referência aos dedos e suas articulações
INDEX_TIP_ID = 8
INDEX_PIP_ID = 6
MIDDLE_TIP_ID = 12
MIDDLE_PIP_ID = 10
RING_TIP_ID = 16
RING_PIP_ID = 14
PINKY_TIP_ID = 20
PINKY_PIP_ID = 18

# responsável pelo comportamento das janelas
def gerenciar_slides_e_janela(image_filenames_list, window_name, win_width, win_height):
    image_paths = []

    # função responsável por criar imagem de placeholder, caso não encontre imagem no caminho especificado
    def create_placeholder_image(filepath, text, width, height):
        # define o placeholder com uma cor aleatória
        img = np.full((height, width, 3), (np.random.randint(50, 150), np.random.randint(50, 150), np.random.randint(50, 150)), dtype=np.uint8)
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        text_x = (width - text_width) // 2
        text_y = (height + text_height) // 2
        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imwrite(filepath, img)
        print(f"Criado placeholder: {filepath}")

    # verifica a existência da imagem, caso não exista, chama a função para criar placeholder
    for i, filename in enumerate(image_filenames_list):
        if not os.path.exists(filename):
            create_placeholder_image(filename, f"Slide de Exemplo {i+1}\n({filename})", win_width, win_height)
        image_paths.append(filename)

    # cria a janela para exibição dos slides
    cv2.namedWindow(window_name)

    # função para exibir o slide atual, recebendo seu index
    def exibir_slide_atual(current_image_index_interno, image_paths_interno):
        # caso o índice da imagem não seja válido ou não há imagens para exibir
        if not image_paths_interno or not (0 <= current_image_index_interno < len(image_paths_interno)):
            error_img = np.full((win_height, win_width, 3), (0, 0, 50), dtype=np.uint8)
            cv2.putText(error_img, "Erro: Imagem nao encontrada", (50, win_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(window_name, error_img) # mostra imagem de erro
            return

        image_path = image_paths_interno[current_image_index_interno] # imagem atual de acordo com indice
        img = cv2.imread(image_path) # carrega imagem

        # caso ocorra um erro ao carregar a imagem, cria imagem de erro
        if img is None:
            error_img = np.full((win_height, win_width, 3), (0, 0, 50), dtype=np.uint8)
            cv2.putText(error_img, f"Erro ao carregar: {os.path.basename(image_path)}", (50, win_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(window_name, error_img)
            return

        # ajusta posição da imagem na janela de acordo com a proporção de ambas
        img_h, img_w = img.shape[:2]
        aspect_ratio_img = img_w / img_h # proporção imagem
        aspect_ratio_win = win_width / win_height # proporção janela

        if aspect_ratio_img > aspect_ratio_win:
            new_w = win_width
            new_h = int(new_w / aspect_ratio_img)
        else:
            new_h = win_height
            new_w = int(new_h * aspect_ratio_img)

        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA) # redimensiona imagem com interpolação para não perder qualidade
        canvas = np.full((win_height, win_width, 3), 0, dtype=np.uint8)
        x_offset = (win_width - new_w) // 2 # centraliza imagem na horizontal
        y_offset = (win_height - new_h) // 2 # centraliza imagem na vertical
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img # insere imagem no canvas

        # mostra slide na janela
        cv2.imshow(window_name, canvas)
        cv2.setWindowTitle(window_name, f"Slide: {os.path.basename(image_path)} ({current_image_index_interno+1}/{len(image_paths_interno)})")

    return image_paths, exibir_slide_atual

def inicializar_componentes_deteccao():
    # 1) Seleciona o módulo de detecção de mãos do MediaPipe
    mp_hands = mp.solutions.hands

    # 2) Cria o “detector” propriamente dito com parâmetros de configuração:
    hands = mp_hands.Hands(
        max_num_hands=2,                # permite detectar até 2 mãos simultaneamente
        min_detection_confidence=0.7,   # confiança mínima (0–1) para considerar uma detecção válida
        min_tracking_confidence=0.7     # confiança mínima (0–1) para continuar rastreando mãos já detectadas
    )

    # 3) Importa utilitários de desenho para sobrepor esqueleto das mãos
    mp_drawing = mp.solutions.drawing_utils
    # 4) Importa estilos de desenho (cores, espessuras etc.)
    mp_drawing_styles = mp.solutions.drawing_styles

    # 5) Inicializa a captura de vídeo (webcam, índice 0)
    cap = cv2.VideoCapture(0)
    # 6) Verifica se a câmera foi aberta corretamente
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        # Se der erro, retorna tupla de Nones para sinalizar falha
        return None, None, None, None

    # 7) Retorna os quatro componentes necessários:
    #    - cap: objeto de captura de vídeo
    #    - hands: detector de mãos configurado
    #    - mp_drawing: utilitários de desenho
    #    - mp_drawing_styles: estilos de desenho
    return cap, hands, mp_drawing, mp_drawing_styles

def detectar_gesto_indicador_levantado(landmarks_pixels):
    # Se não houver exatamente 21 pontos da mão, não é um conjunto válido → gesto não detectado
    if len(landmarks_pixels) != 21:
        return False

    # Compara a altura (y) da ponta com articulação do indicador, tip = ponta, pip = articulação
    index_finger_up  = landmarks_pixels[INDEX_TIP_ID][1]  < landmarks_pixels[INDEX_PIP_ID][1]
    # Compara a altura do dedo médio: True se estiver dobrado/baixo
    middle_finger_down = landmarks_pixels[MIDDLE_TIP_ID][1] >= landmarks_pixels[MIDDLE_PIP_ID][1]
    # Mesmo para o dedo anelar
    ring_finger_down   = landmarks_pixels[RING_TIP_ID][1]   >= landmarks_pixels[RING_PIP_ID][1]
    # Mesmo para o dedo mínimo
    pinky_finger_down  = landmarks_pixels[PINKY_TIP_ID][1]  >= landmarks_pixels[PINKY_PIP_ID][1]

    # Retorna True somente se indicador está para cima e todos os outros três estão para baixo
    return (index_finger_up
            and middle_finger_down
            and ring_finger_down
            and pinky_finger_down)

def processar_entrada_e_reconhecer_gestos(
    hands_results,        # resultado da detecção de mãos pelo MediaPipe
    image_camera,         # frame capturado da câmera (BGR)
    current_time_sec,     # timestamp atual (em segundos)
    last_action_time_sec, # timestamp da última ação realizada
    current_slide_idx,    # índice do slide que está sendo exibido
    all_image_paths,      # lista com todos os caminhos das imagens dos slides
    fn_exibir_slide,      # função para mostrar um slide na janela
    mp_hands_module,      # módulo mp.solutions.hands
    mp_drawing_module,    # módulo mp.solutions.drawing_utils
    mp_drawing_styles_module # módulo mp.solutions.drawing_styles
):

    # Texto que será desenhado na tela mostrando qual gesto foi reconhecido
    gesture_feedback_text = ""

    # Flag para garantir só UMA ação por frame (evita avançar e voltar no mesmo loop)
    action_taken_this_frame = False

    # Só executa se houver landmarks de mão E informação de “handedness” (direita/esquerda)
    if hands_results.multi_hand_landmarks and hands_results.multi_handedness:

        # Itera sobre cada mão detectada
        for i, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):

            # Se já fizemos uma ação neste frame, saímos do loop
            if action_taken_this_frame:
                break

            # Desenha os pontos e conexões da mão sobre o frame
            mp_drawing_module.draw_landmarks(
                image_camera,                    # frame onde desenhar
                hand_landmarks,                  # pontos da mão
                mp_hands_module.HAND_CONNECTIONS,# conexões entre pontos
                mp_drawing_styles_module.get_default_hand_landmarks_style(),
                mp_drawing_styles_module.get_default_hand_connections_style()
            )

            # Recupera a informação de “left” ou “right” para essa mão
            handedness_info = hands_results.multi_handedness[i]
            hand_label = handedness_info.classification[0].label
            # → 'Right' ou 'Left'

            # Converte cada landmark (normalizado) para coordenada em pixels
            h, w, _ = image_camera.shape
            landmarks_pixels = []
            for lm in hand_landmarks.landmark:
                # lm.x e lm.y variam de 0–1, então multiplica pelo tamanho da imagem
                landmarks_pixels.append((int(lm.x * w), int(lm.y * h)))

            # ───────────────────────────────────────────────────────────────────
            # Aqui chamamos a função que checa se **apenas** o indicador está reto
            is_pointing = detectar_gesto_indicador_levantado(landmarks_pixels)
    
            # verifica se o gesto está sendo feito e se está fora do cooldown
            if is_pointing and (current_time_sec - last_action_time_sec > ACTION_COOLDOWN):
                action_performed_by_this_hand = False # variável auxiliar para identificar a mão
                
                if hand_label == "Right": # verifica se é a mão direita
                    if current_slide_idx < len(all_image_paths) - 1: # verifica se não é o último slide
                        current_slide_idx += 1 # incrementa o indice dos slides
                        fn_exibir_slide(current_slide_idx, all_image_paths) # vai exibir o slide
                        gesture_feedback_text = "AVANCAR (Direita)" # mensagem que vai ser exibida na câmera
                        last_action_time_sec = current_time_sec # atualiza o timestamp
                        action_performed_by_this_hand = True # indica que a ação foi performada
                    else:
                        gesture_feedback_text = "FIM DOS SLIDES" # caso seja o último slide, indica que é o fim deles
                        last_action_time_sec = current_time_sec
                
                elif hand_label == "Left": # verifica se é a mão esquerda
                    if current_slide_idx > 0: # verifica se não é o primeiro slide
                        current_slide_idx -= 1 # decrementa o índice dos slides
                        fn_exibir_slide(current_slide_idx, all_image_paths)
                        gesture_feedback_text = "VOLTAR (Esquerda)"
                        last_action_time_sec = current_time_sec
                        action_performed_by_this_hand = True
                    else:
                        gesture_feedback_text = "INICIO DOS SLIDES" # caso seja o primeiro, indica que é o início deles
                        last_action_time_sec = current_time_sec
                
                if action_performed_by_this_hand: # se alguma mão executou alguma ação
                    action_taken_this_frame = True # informa que uma ação já foi executada no frame
                    print(gesture_feedback_text)
    # se houver uma mensagem de feedback
    if gesture_feedback_text:
        feedback_color = (255, 255, 0) # padrão de cor para feedback = amarelo
        if "AVANCAR" in gesture_feedback_text: feedback_color = (0, 255, 0) # cor de avançar = verde
        elif "VOLTAR" in gesture_feedback_text: feedback_color = (0, 0, 255) # cor de retornar = azul
        
        cv2.putText(image_camera, gesture_feedback_text, (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    2, feedback_color, 3) # adiciona as mensagens à camera

    # retorna o indíce do slide atual, timestamp da última ação e o frame da câmera já com o feedback sobreposto
    return current_slide_idx, last_action_time_sec, image_camera

def main():
    image_paths, exibir_slide_atual_fn = gerenciar_slides_e_janela(
        IMAGE_FILENAMES, SLIDE_WINDOW_NAME, SLIDE_WINDOW_WIDTH, SLIDE_WINDOW_HEIGHT
    )
    current_image_index = 0
    if not image_paths:
        print("Nenhuma imagem para exibir. Saindo.")
        return
    exibir_slide_atual_fn(current_image_index, image_paths)

    cap, hands_detector, mp_drawing, mp_drawing_styles = inicializar_componentes_deteccao()
    if not cap:
        cv2.destroyAllWindows()
        return

    last_action_time = 0
    print("Câmera iniciada. Mostre os gestos para a câmera.")
    print(f"  - MÃO DIREITA, dedo indicador levantado e outros abaixados: Avançar slide")
    print(f"  - MÃO ESQUERDA, dedo indicador levantado e outros abaixados: Voltar slide")
    print("Pressione 'q' na janela da câmera para sair.")

    while cap.isOpened():
        success, image_from_camera = cap.read()
        if not success:
            continue

        image_rgb = cv2.cvtColor(cv2.flip(image_from_camera, 1), cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands_detector.process(image_rgb)
        image_bgr_display = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        image_bgr_display.flags.writeable = True
        current_time = time.time()

        current_image_index, last_action_time, image_with_feedback = processar_entrada_e_reconhecer_gestos(
            results, image_bgr_display, current_time, last_action_time,
            current_image_index, image_paths, exibir_slide_atual_fn,
            mp.solutions.hands, mp_drawing, mp_drawing_styles
        )
        
        cv2.imshow('Controle de Gestos - Camera (Pressione "q" para sair)', image_with_feedback)

        if cv2.getWindowProperty(SLIDE_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print(f"Janela '{SLIDE_WINDOW_NAME}' fechada. Saindo.")
            break
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if hands_detector:
        hands_detector.close()
    print("Programa finalizado.")

if __name__ == '__main__':
    main()