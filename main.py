import cv2
import mediapipe as mp
import time
import numpy as np
import os

# --- Configurações Globais ---
# CRIE ESTAS IMAGENS NA MESMA PASTA DO SCRIPT, OU ELAS SERÃO GERADAS AUTOMATICAMENTE
IMAGE_FILENAMES = [
    "slides/slide1.png",
    "slides/slide2.png",
    "slides/slide3.png",
    "slides/slide4.png",
    "slides/slide5.png"
]
SLIDE_WINDOW_WIDTH = 800
SLIDE_WINDOW_HEIGHT = 600
SLIDE_WINDOW_NAME = "Slide Atual"

# Configurações de Gestos
ACTION_COOLDOWN = 1.5  # Segundos
INDEX_TIP_ID = 8
INDEX_PIP_ID = 6
MIDDLE_TIP_ID = 12
MIDDLE_PIP_ID = 10
RING_TIP_ID = 16
RING_PIP_ID = 14
PINKY_TIP_ID = 20
PINKY_PIP_ID = 18

# --- FUNÇÃO 1: Gerenciamento de Slides e Janela ---
#  (como a parte que lida com a apresentação visual dos slides)
def gerenciar_slides_e_janela(image_filenames_list, window_name, win_width, win_height):
    """
    Inicializa os caminhos das imagens, cria placeholders se necessário,
    cria a janela de slides e retorna uma função para exibir os slides.
    """
    image_paths = []

    def create_placeholder_image(filepath, text, width, height):
        img = np.full((height, width, 3), (np.random.randint(50, 150), np.random.randint(50, 150), np.random.randint(50, 150)), dtype=np.uint8)
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        text_x = (width - text_width) // 2
        text_y = (height + text_height) // 2
        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imwrite(filepath, img)
        print(f"Criado placeholder: {filepath}")

    for i, filename in enumerate(image_filenames_list):
        if not os.path.exists(filename):
            create_placeholder_image(filename, f"Slide de Exemplo {i+1}\n({filename})", win_width, win_height)
        image_paths.append(filename)

    cv2.namedWindow(window_name)

    def exibir_slide_atual(current_image_index_interno, image_paths_interno):
        if not image_paths_interno or not (0 <= current_image_index_interno < len(image_paths_interno)):
            error_img = np.full((win_height, win_width, 3), (0, 0, 50), dtype=np.uint8)
            cv2.putText(error_img, "Erro: Imagem nao encontrada", (50, win_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(window_name, error_img)
            return

        image_path = image_paths_interno[current_image_index_interno]
        img = cv2.imread(image_path)

        if img is None:
            error_img = np.full((win_height, win_width, 3), (0, 0, 50), dtype=np.uint8)
            cv2.putText(error_img, f"Erro ao carregar: {os.path.basename(image_path)}", (50, win_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(window_name, error_img)
            return

        img_h, img_w = img.shape[:2]
        aspect_ratio_img = img_w / img_h
        aspect_ratio_win = win_width / win_height

        if aspect_ratio_img > aspect_ratio_win:
            new_w = win_width
            new_h = int(new_w / aspect_ratio_img)
        else:
            new_h = win_height
            new_w = int(new_h * aspect_ratio_img)

        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((win_height, win_width, 3), 0, dtype=np.uint8)
        x_offset = (win_width - new_w) // 2
        y_offset = (win_height - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img

        cv2.imshow(window_name, canvas)
        cv2.setWindowTitle(window_name, f"Slide: {os.path.basename(image_path)} ({current_image_index_interno+1}/{len(image_paths_interno)})")

    return image_paths, exibir_slide_atual

# --- FUNÇÃO 2: Inicialização dos Componentes de Detecção ---
# (configuração do MediaPipe Hands) Rian
def inicializar_componentes_deteccao():
    """Inicializa a câmera e o MediaPipe Hands."""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return None, None, None, None
    return cap, hands, mp_drawing, mp_drawing_styles

# --- FUNÇÃO 3: Detecção do Gesto de "Indicador Levantado" ---
# identificação da pose da mão
def detectar_gesto_indicador_levantado(landmarks_pixels):
    """
    Verifica se o dedo indicador está levantado e os outros dedos principais estão abaixados.
    Retorna True se o gesto for detectado, False caso contrário.
    """
    if len(landmarks_pixels) != 21:
        return False

    index_finger_up = landmarks_pixels[INDEX_TIP_ID][1] < landmarks_pixels[INDEX_PIP_ID][1]
    middle_finger_down = landmarks_pixels[MIDDLE_TIP_ID][1] >= landmarks_pixels[MIDDLE_PIP_ID][1]
    ring_finger_down = landmarks_pixels[RING_TIP_ID][1] >= landmarks_pixels[RING_PIP_ID][1]
    pinky_finger_down = landmarks_pixels[PINKY_TIP_ID][1] >= landmarks_pixels[PINKY_PIP_ID][1]

    return index_finger_up and middle_finger_down and ring_finger_down and pinky_finger_down

# --- FUNÇÃO 4: Processamento de Entrada e Reconhecimento de Gestos para Ações ---
# Esta função será dividida na explicação entre Rian e Humberto.
def processar_entrada_e_reconhecer_gestos(
    hands_results, image_camera, current_time_sec, last_action_time_sec,
    current_slide_idx, all_image_paths, fn_exibir_slide,
    mp_hands_module, mp_drawing_module, mp_drawing_styles_module):
    """
    Processa os resultados da detecção de mãos, identifica gestos e atualiza os slides.
    Retorna o novo índice do slide, o novo tempo da última ação e o frame da câmera com desenhos.
    """
    gesture_feedback_text = ""
    action_taken_this_frame = False

    if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
        for i, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            if action_taken_this_frame:
                break

            # ##--- INÍCIO DA EXPLICAÇÃO - Rian ---##
            # Parte: Processamento dos dados brutos da mão e preparação para detecção do gesto.

            # Desenhar landmarks para cada mão
            mp_drawing_module.draw_landmarks(
                image_camera, hand_landmarks, mp_hands_module.HAND_CONNECTIONS,
                mp_drawing_styles_module.get_default_hand_landmarks_style(),
                mp_drawing_styles_module.get_default_hand_connections_style()
            )

            handedness_info = hands_results.multi_handedness[i]
            hand_label = handedness_info.classification[0].label

            h, w, _ = image_camera.shape
            landmarks_pixels = []
            for lm in hand_landmarks.landmark:
                landmarks_pixels.append((int(lm.x * w), int(lm.y * h)))

            # Chamada para a Função 3 (detectar_gesto_indicador_levantado),
            # que Rian explicou em detalhe anteriormente.
            # O resultado 'is_pointing' é a identificação de baixo nível do gesto.
            is_pointing = detectar_gesto_indicador_levantado(landmarks_pixels)
            # ##--- FIM DA EXPLICAÇÃO - Rian ---##


            # (Rian passa a explicação para Humberto neste ponto)


            # ##--- INÍCIO DA EXPLICAÇÃO - Humberto ---##
            # Parte: Tomada de decisão e ação com base no gesto identificado e na mão.

            if is_pointing and (current_time_sec - last_action_time_sec > ACTION_COOLDOWN):
                action_performed_by_this_hand = False
                
                if hand_label == "Right":
                    if current_slide_idx < len(all_image_paths) - 1:
                        current_slide_idx += 1
                        fn_exibir_slide(current_slide_idx, all_image_paths)
                        gesture_feedback_text = "AVANCAR (Direita)"
                        last_action_time_sec = current_time_sec
                        action_performed_by_this_hand = True
                    else:
                        gesture_feedback_text = "FIM DOS SLIDES"
                        last_action_time_sec = current_time_sec
                
                elif hand_label == "Left":
                    if current_slide_idx > 0:
                        current_slide_idx -= 1
                        fn_exibir_slide(current_slide_idx, all_image_paths)
                        gesture_feedback_text = "VOLTAR (Esquerda)"
                        last_action_time_sec = current_time_sec
                        action_performed_by_this_hand = True
                    else:
                        gesture_feedback_text = "INICIO DOS SLIDES"
                        last_action_time_sec = current_time_sec
                
                if action_performed_by_this_hand:
                    action_taken_this_frame = True
                    print(gesture_feedback_text) # Mover o print para quando a ação é realmente feita

    # Desenhar feedback na tela da câmera se houver (Humberto continua explicando)
    if gesture_feedback_text:
        feedback_color = (255, 255, 0)
        if "AVANCAR" in gesture_feedback_text: feedback_color = (0, 255, 0)
        elif "VOLTAR" in gesture_feedback_text: feedback_color = (0, 0, 255)
        
        cv2.putText(image_camera, gesture_feedback_text, (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    2, feedback_color, 3)
    # ##--- FIM DA EXPLICAÇÃO - Humberto ---##

    return current_slide_idx, last_action_time_sec, image_camera


# --- FUNÇÃO PRINCIPAL (main) ---
def main():
    # Gerenciar slides e janela (Pode ser mencionado brevemente no início)
    image_paths, exibir_slide_atual_fn = gerenciar_slides_e_janela(
        IMAGE_FILENAMES, SLIDE_WINDOW_NAME, SLIDE_WINDOW_WIDTH, SLIDE_WINDOW_HEIGHT
    )
    current_image_index = 0
    if not image_paths:
        print("Nenhuma imagem para exibir. Saindo.")
        return
    exibir_slide_atual_fn(current_image_index, image_paths)

    # Inicializar componentes de detecção (Rian começa aqui sua explicação detalhada)
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
        results = hands_detector.process(image_rgb) # (Rian explica o que acontece aqui)
        image_bgr_display = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        image_bgr_display.flags.writeable = True
        current_time = time.time()

        # Processar entrada e reconhecer gestos (Rian e Humberto dividem a explicação desta função)
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