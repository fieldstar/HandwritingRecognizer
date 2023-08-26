# %load app.py
"""
@author: 曾小青<zengxq@csust.edu.cn>
"""
import os
import random 
from flask import Flask, request, render_template
import csv
import pandas as pd
import numpy as np  
import Recognizer as recog

app = Flask(__name__,static_folder='data', static_url_path='/static')

# 文件上传目录
app.config['UPLOAD_FOLDER'] = 'data/uploads/'
app.config['TEST_FOLDER'] = 'data/test/'
# 支持的文件格式
app.config['ALLOWED_EXTENSIONS'] = {'PNG','JPG','JPEG','GIF','png','jpg', 'jpeg', 'gif'}  # 集合类型
 
# 判断文件名是否是我们支持的格式
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

df_word_pinyin=pd.read_csv('data/word_pinyin.csv',header=0,encoding='gbk')
dict_word_pinyin=df_word_pinyin[["word", "pinyin"]].set_index("word").to_dict(orient='dict')["pinyin"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/input")
def new_input():
    return render_template("input.html")

@app.route("/show", methods=["POST"])
def show():
    upload_file = request.files['image_name']
    if upload_file and allowed_file(upload_file.filename):
        filename = upload_file.filename
        # 将文件保存到 static/uploads 目录，文件名同上传时使用的文件名
        full_img_path=os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
        upload_file.save(full_img_path)
        word,label=recog.predict_by_image_name(full_img_path,recog.model_0)
        word_writer,label_writer=recog.predict_writer_by_image_name(full_img_path,recog.writer_model_0)
        return render_template("result.html", filename=filename,word=word,writer=word_writer,pinyin=dict_word_pinyin[word]) 
    else:
        return '图像文件：'+filename+'上传失败，请检查文件否为(jpg,gif,png)类型...'



@app.route("/word_list", methods=["POST"])
def word_list():
    word=request.form['word']
    images_name_list=recog.get_images_path_by_word(word)
    
    return render_template("word_list.html",word=word,pinyin=dict_word_pinyin[word],name_list=images_name_list,base_url='/static/wordlib/')

@app.route("/writer_list", methods=["POST"])
def writer_list():
    writer=request.form['writer']
    images_name_list=recog.get_images_path_by_writer(writer)
    
    return render_template("writer_list.html",writer=writer,name_list=images_name_list,base_url='/static/wordlib/')


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(host='0.0.0.0',port=5555, debug=False)