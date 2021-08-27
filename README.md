# compressProject
Here is the main idea of total architexture design in my work.The left is the encoder which can map the original file to a latent space, which size is smaller than that of original file.The middle nodes represent the compressed code; the right is the decode, by which the file recovers from the compressed code.

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190227162914381.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9zaWdhaS5ibG9nLmNzZG4ubmV0,size_16,color_FFFFFF,t_70)

We assume that the file size is 10KB.

the encoders's structure is following:

1. the filst layer is **convolution** at 1 dimension whose kernal size is 12289 with 10 channels .
2. the second layer is **convolution** at 1 dimension whose kernal size is 4097 with 10 channels .
3. the third,forth, fifth layers are **fully connection** with 2048, 1024,512 output feature number respectively. The activation  function  of them is **RELU** function.

the decode's structure is following:

there are 6 layers,all of which are fully connection. their output feature number  are 1024, 2048, 4096,8192,8192,10240 respectively. The activation functions of the first four layers   are **RELU**, whereas the counterpart of left layer is **tanh**. Batchnorm is used between each layer.

I use the root mean square error of the restored file content and the original file content as the cost function.