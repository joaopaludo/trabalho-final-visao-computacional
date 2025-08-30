import cv2
import os

# --- CONFIGURAÇÕES ---
input_folder = "moedas/moedas_treino"
output_folder = "moedas_redimensionadas"
target_size = (450, 450)


if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Diretório de saída criado em: {output_folder}")

class_folders = os.listdir(input_folder)

for class_name in class_folders:
    input_class_path = os.path.join(input_folder, class_name)

    if not os.path.isdir(input_class_path):
        continue

    side_folders = os.listdir(input_class_path)
    for side_name in side_folders:
        input_side_path = os.path.join(input_class_path, side_name)

        if not os.path.isdir(input_side_path):
            continue

        output_side_path = os.path.join(output_folder, class_name, side_name)
        if not os.path.exists(output_side_path):
            os.makedirs(output_side_path)

        print(f"\nProcessando: Classe '{class_name}', Lado '{side_name}'")

        image_files = os.listdir(input_side_path)
        
        image_count = 0

        for filename in image_files:
            input_image_path = os.path.join(input_side_path, filename)
            output_image_path = os.path.join(output_side_path, filename)
            
            try:
                image = cv2.imread(input_image_path)

                if image is not None:
                    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

                    cv2.imwrite(output_image_path, resized_image)
                    image_count += 1
                else:
                    print(f" Não foi possível ler o arquivo: {filename}")

            except Exception as e:
                print(f"[Erro] Ocorreu um erro ao processar o arquivo {filename}: {e}")

        print(f" -> {image_count} imagens redimensionadas e salvas em '{output_side_path}'")

print("\nProcesso concluído! Seu novo dataset está pronto em:", output_folder)
