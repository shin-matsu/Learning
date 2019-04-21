from PIL import Image
import sys
import cv2
import pyocr
import pyocr.builders

#利用可能なツールのリストを取得
tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)

print(tools)

tool = tools[0]
print("Will use tool: '%s'" % (tool.get_name()))

args = sys.argv
input_image = args[1]

print("-------------------")
print("画像：" + input_image)
print("-------------------")

#利用可能な言語の確認
langs = tool.get_available_languages()
print("利用可能な言語: %s" % ", ".join(langs))

#img_src = cv2.imread("./" + input_image, 1)
#input_image = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

txt = tool.image_to_string(
    Image.open(input_image),
    lang="jpn",
    builder=pyocr.builders.TextBuilder(tesseract_layout=6)
)
print( txt )

res = tool.image_to_string(
    Image.open(input_image),
    lang="jpn",
    builder=pyocr.builders.WordBoxBuilder(tesseract_layout=6)
)

out = cv2.imread(input_image)
for d in res:
    print(d.content)
    print(d.position)
    cv2.rectangle(out, d.position[0], d.position[1], (0, 0, 255), 2)

cv2.imshow("img",out)
cv2.waitKey(0)
cv2.destroyAllWindows()