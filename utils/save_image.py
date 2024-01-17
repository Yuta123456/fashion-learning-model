from PIL import Image, ImageDraw, ImageFont


def save_combined_image(item_ids, output_path, W=3):
    # 画像を3×3に結合するための準備
    images = [
        Image.open(f"C:/Users/yuuta/Documents/fashion/data/images/{item_id}.jpg")
        for item_id in item_ids
    ]

    H = len(images) // W
    width, height = images[0].size
    result_image = Image.new("RGB", (width * W, height * H))
    print(W, H, f"size: ({width * W}, {height * H})")
    for i in range(H):
        for j in range(W):
            index = i * W + j
            result_image.paste(images[index], (j * width, i * height))

    # 結合した画像を保存
    result_image.save(output_path)
    return result_image
