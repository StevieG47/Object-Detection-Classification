import tensorflow as tf


def getBreedLabels(labelsFile):
    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                       in tf.gfile.GFile(labelsFile)]
    return label_lines


def getBreed(im, label_lines, model):
        
    # Read in the image_data
   # image_data = tf.gfile.FastGFile(im, 'rb').read()
    
   
    # Unpersists graph from file
    with tf.gfile.FastGFile(model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    
    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg:0': im})
    
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
        #for node_id in top_k:
        #    human_string = label_lines[node_id]
        #    score = predictions[0][node_id]
        #    print('%s (score = %.5f)' % (human_string, score))
    
    
    # Get predicted class
    predicted_breed = label_lines[top_k[0]]
    
    return predicted_breed
    
