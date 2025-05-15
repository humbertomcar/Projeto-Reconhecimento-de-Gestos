import cv2
import mediapipe as mp
import time
import numpy as np
import os

# --- Configurações das Imagens/Slides ---
# CRIE ESTAS IMAGENS NA MESMA PASTA DO SCRIPT, OU ELAS SERÃO GERADAS AUTOMATICAMENTE
IMAGE_FILENAMES = [
    "slide1.png",
    "slide2.png",
    "slide3.png",
    "slide4.png",
    "slide5.png"
]
IMAGE_PATHS = [] # Será preenchido abaixo

# Dimensões para os slides/imagens exibidas
SLIDE_WINDOW_WIDTH = 800
SLIDE_WINDOW_HEIGHT = 600
SLIDE_WINDOW_NAME = "Slide Atual"

# --- Geração de Placeholders (se as imagens não existirem) ---
def create_placeholder_image(filepath, text):
    img = np.full((SLIDE_WINDOW_HEIGHT, SLIDE_WINDOW_WIDTH, 3), (np.random.randint(50,150), np.random.randint(50,150), np.random.randint(50,150)), dtype=np.uint8)
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    text_x = (SLIDE_WINDOW_WIDTH - text_width) // 2
    text_y = (SLIDE_WINDOW_HEIGHT + text_height) // 2
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.imwrite(filepath, img)
    print(f"Criado placeholder: {filepath}")

for i, filename in enumerate(IMAGE_FILENAMES):
    if not os.path.exists(filename):
        create_placeholder_image(filename, f"Slide de Exemplo {i+1}\n({filename})")
    IMAGE_PATHS.append(filename)

current_image_index = 0

# --- Função para exibir o slide/imagem atual ---
def update_slide_display():
    if not IMAGE_PATHS or not (0 <= current_image_index < len(IMAGE_PATHS)):
        error_img = np.full((SLIDE_WINDOW_HEIGHT, SLIDE_WINDOW_WIDTH, 3), (0,0,50), dtype=np.uint8)
        cv2.putText(error_img, "Erro: Imagem nao encontrada", (50, SLIDE_WINDOW_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(SLIDE_WINDOW_NAME, error_img)
        return

    image_path = IMAGE_PATHS[current_image_index]
    img = cv2.imread(image_path)

    if img is None:
        error_img = np.full((SLIDE_WINDOW_HEIGHT, SLIDE_WINDOW_WIDTH, 3), (0,0,50), dtype=np.uint8)
        cv2.putText(error_img, f"Erro ao carregar: {os.path.basename(image_path)}", (50, SLIDE_WINDOW_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(SLIDE_WINDOW_NAME, error_img)
        return

    img_h, img_w = img.shape[:2]
    aspect_ratio_img = img_w / img_h
    aspect_ratio_win = SLIDE_WINDOW_WIDTH / SLIDE_WINDOW_HEIGHT

    if aspect_ratio_img > aspect_ratio_win:
        new_w = SLIDE_WINDOW_WIDTH
        new_h = int(new_w / aspect_ratio_img)
    else:
        new_h = SLIDE_WINDOW_HEIGHT
        new_w = int(new_h * aspect_ratio_img)

    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((SLIDE_WINDOW_HEIGHT, SLIDE_WINDOW_WIDTH, 3), 0, dtype=np.uint8)
    x_offset = (SLIDE_WINDOW_WIDTH - new_w) // 2
    y_offset = (SLIDE_WINDOW_HEIGHT - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img

    cv2.imshow(SLIDE_WINDOW_NAME, canvas)
    cv2.setWindowTitle(SLIDE_WINDOW_NAME, f"Slide: {os.path.basename(image_path)} ({current_image_index+1}/{len(IMAGE_PATHS)})")


# --- Configurações de Gestos ---
ACTION_COOLDOWN = 1.5  # Segundos
last_action_time = 0
INDEX_TIP_ID = 8
INDEX_PIP_ID = 6
# Para tornar o gesto de "apontar" mais claro, verificaremos se os outros dedos estão abaixados
MIDDLE_TIP_ID = 12
MIDDLE_PIP_ID = 10
RING_TIP_ID = 16
RING_PIP_ID = 14
PINKY_TIP_ID = 20
PINKY_PIP_ID = 18

# --- Inicialização do MediaPipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,  # <<< MODIFICADO: Detectar até 2 mãos
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Inicialização da Câmera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera.")
    exit()

# --- Criar janela para os slides e exibir o primeiro ---
cv2.namedWindow(SLIDE_WINDOW_NAME)
update_slide_display()

print("Câmera iniciada. Mostre os gestos para a câmera.")
print(f"  - MÃO DIREITA, dedo indicador levantado: Avançar slide")
print(f"  - MÃO ESQUERDA, dedo indicador levantado: Voltar slide")
print("Pressione 'q' na janela da câmera para sair.")

# --- Loop Principal ---
while cap.isOpened():
    success, image_camera = cap.read()
    if not success:
        print("Ignorando frame vazio da câmera.")
        continue

    image_camera = cv2.cvtColor(cv2.flip(image_camera, 1), cv2.COLOR_BGR2RGB)
    image_camera.flags.writeable = False
    results = hands.process(image_camera)
    image_camera.flags.writeable = True
    image_camera = cv2.cvtColor(image_camera, cv2.COLOR_RGB2BGR)

    current_time = time.time()
    gesture_text_feedback = ""

    if results.multi_hand_landmarks and results.multi_handedness: # <<< MODIFICADO: Checar também multi_handedness
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Desenhar landmarks para cada mão
            mp_drawing.draw_landmarks(
                image_camera,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Obter informações de qual mão é (Esquerda/Direita)
            handedness_info = results.multi_handedness[i]
            hand_label = handedness_info.classification[0].label # Será 'Left' ou 'Right'
            # hand_score = handedness_info.classification[0].score # Confiança da predição

            h, w, _ = image_camera.shape
            landmarks_pixels = []
            for lm in hand_landmarks.landmark:
                landmarks_pixels.append((int(lm.x * w), int(lm.y * h)))

            if len(landmarks_pixels) == 21:
                # Verificar se o dedo indicador está levantado
                index_finger_up = landmarks_pixels[INDEX_TIP_ID][1] < landmarks_pixels[INDEX_PIP_ID][1]
                # Para um gesto mais claro, verificar se os outros dedos estão abaixados
                middle_finger_down = landmarks_pixels[MIDDLE_TIP_ID][1] >= landmarks_pixels[MIDDLE_PIP_ID][1]
                ring_finger_down = landmarks_pixels[RING_TIP_ID][1] >= landmarks_pixels[RING_PIP_ID][1]
                pinky_finger_down = landmarks_pixels[PINKY_TIP_ID][1] >= landmarks_pixels[PINKY_PIP_ID][1]

                # Definir o gesto de "apontar" (indicador para cima, outros para baixo)
                is_pointing_gesture = index_finger_up and middle_finger_down and ring_finger_down and pinky_finger_down

                if current_time - last_action_time > ACTION_COOLDOWN:
                    action_performed = False
                    
                    # Gesto: Avançar (Mão Direita, indicador levantado)
                    if hand_label == "Right" and is_pointing_gesture:
                        if current_image_index < len(IMAGE_PATHS) - 1:
                            current_image_index += 1
                            update_slide_display()
                            gesture_text_feedback = "AVANCAR (Direita)"
                            print(gesture_text_feedback)
                            last_action_time = current_time
                            action_performed = True
                        else:
                            gesture_text_feedback = "FIM DOS SLIDES"
                            print(gesture_text_feedback)
                            last_action_time = current_time # Cooldown mesmo no limite
                    
                    # Gesto: Voltar (Mão Esquerda, indicador levantado)
                    elif hand_label == "Left" and is_pointing_gesture:
                        if current_image_index > 0:
                            current_image_index -= 1
                            update_slide_display()
                            gesture_text_feedback = "VOLTAR (Esquerda)"
                            print(gesture_text_feedback)
                            last_action_time = current_time
                            action_performed = True
                        else:
                            gesture_text_feedback = "INICIO DOS SLIDES"
                            print(gesture_text_feedback)
                            last_action_time = current_time # Cooldown mesmo no limite
                    
                    if action_performed or "SLIDES" in gesture_text_feedback:
                        # Determinar a cor do feedback
                        feedback_color = (255, 255, 0) # Amarelo para "INICIO/FIM"
                        if "AVANCAR" in gesture_text_feedback:
                            feedback_color = (0, 255, 0) # Verde para avançar
                        elif "VOLTAR" in gesture_text_feedback:
                            feedback_color = (0, 0, 255) # Vermelho para voltar
                        
                        cv2.putText(image_camera, gesture_text_feedback, (20, 70), cv2.FONT_HERSHEY_PLAIN,
                                    2, feedback_color, 3)
                        # Se uma ação foi realizada, podemos quebrar o loop das mãos,
                        # pois não queremos processar a outra mão no mesmo frame para uma ação de slide.
                        # Isso evita que, se ambas as mãos fizerem o gesto ao mesmo tempo, ele tente avançar e voltar.
                        if action_performed:
                            break # Sai do loop 'for hand_landmarks...'


    cv2.imshow('Controle de Gestos - Camera (Pressione "q" para sair)', image_camera)

    if cv2.getWindowProperty(SLIDE_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        print(f"Janela '{SLIDE_WINDOW_NAME}' fechada. Saindo.")
        break

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("Programa finalizado.")