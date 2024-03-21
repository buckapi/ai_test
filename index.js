import { config } from 'dotenv';
config();

import { HfInference } from '@huggingface/inference';
import readline from 'readline';

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

async function getImageCaption() {
    const hf = new HfInference(process.env.HUGGINGFACE_ACCESS_TOKEN); // Accede al access token desde el entorno
    
    rl.question('Introduce la URL de la imagen: ', async (imageUrl) => {
        try {
            const response = await fetch(imageUrl);
            const model = "Salesforce/blip-image-captioning-large";
            const blob = await response.blob();
            
            try {
                const result = await hf.imageToText({
                    data: blob,
                    model: model
                });
                console.log("Texto detectado en la imagen:", result);
                
                try {
                    const translatedResult = await hf.translation({
                        model: 'facebook/mbart-large-50-many-to-many-mmt',
                        inputs: result.generated_text,
                        parameters: {
                            "src_lang": "en_XX",
                            "tgt_lang": "es_XX"
                        }
                    });
                    console.log("Texto traducido:", translatedResult);
                } catch (error) {
                    console.error("Error al traducir el texto:", error);
                }
                
            } catch (error) {
                console.error("Error al convertir la imagen a texto:", error);
            }
        } catch (error) {
            console.error("Error al obtener la imagen:", error);
        } finally {
            rl.close(); // Cerramos la interfaz de lectura una vez que se completa la operación
        }
    });
}

getImageCaption();
