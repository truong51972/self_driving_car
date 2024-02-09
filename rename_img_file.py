import os

def list_jpg_files_in_directory(directory_path):
    file_list = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.jpg'):
                file_list.append(os.path.join(root, file))
    return file_list

def rename_file(old_name, new_name):
    try:
        os.rename(old_name, new_name)
        print(f"Đã đổi tên từ '{old_name}' thành '{new_name}'.")
    except OSError:
        print(f"Lỗi: Không thể đổi tên từ '{old_name}' thành '{new_name}'.")


def create_directory(directory_path):
    try:
        # Tạo thư mục với đường dẫn đã chỉ định
        os.mkdir(directory_path)
        print(f"Đã tạo thành công thư mục: {directory_path}.")
    except FileExistsError:
        print(f"Thư mục {directory_path} đã tồn tại.")
    except Exception as e:
        print(f"Có lỗi xảy ra khi tạo thư mục: {e}")


directory_path = "./imgs"

file_list = list_jpg_files_in_directory(directory_path)

for i, file_path in enumerate(file_list):
    # folder_name = f'part{int(round(i/12,0))}/'
    # folder_name = f'part{int(round(i/12,0))}/'
    # create_directory(folder_name)
    # rename_file(file_path, f'{folder_name}/{i}.jpg')
    rename_file(file_path, f'{directory_path}/img_{i}.jpg')
    
    # print(int(round(i/12,0)))
    # print(file_path)