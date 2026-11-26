package io.teknek.reckoning.integration.spark;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.generator.GeneratorParameters;
import io.teknek.reckoning.generator.Response;
import io.teknek.reckoning.integration.gemma2.Gemma2Suite;
import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.AutoModelForCausaLm;
import io.teknek.reckoning.model.DoNothingGenerateEvent;
import io.teknek.reckoning.model.RopeStorageMode;
import io.teknek.reckoning.safetensors.fetch.ModelFetcher;
import io.teknek.reckoning.safetensors.prompt.PromptContext;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class SparkPromptIT {

    @Test
    public void gemmaAlbanyFirstTokens() {
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(
                new ModelFetcher("edwardcapriolo", "Spark-X2.6-1.7B-JQ4") );
        builder.withRopeStorageMode(RopeStorageMode.ON_DEMAND);
        //builder.withKvKeyDType(DType.I8);
        //builder.withKvValueDType(DType.I8);
        //builder.withOutputHeadQuantization(DType.Q4);
        AbstractModel model = builder.buildLocalTransformerModel();


        //AbstractModel model = Gemma2Suite.getOrCreate();
        PromptContext prompt = model.promptSupport().orElseThrow().builder()
                .addUserMessage("What is the capital of New York, USA?").build();
        Response response = model.generate(UUID.randomUUID(), prompt, new GeneratorParameters()
                        .withMaxTokens(3)
                        .withTemperature(0.0f)

                        ,
                (a,b,c,d)-> { System.out.print(b);});
        System.out.println(response);
    }
}
