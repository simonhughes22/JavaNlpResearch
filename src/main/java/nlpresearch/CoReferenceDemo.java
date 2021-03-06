package nlpresearch;

import edu.stanford.nlp.coref.CorefCoreAnnotations;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.coref.data.Mention;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

import java.util.Properties;

public class CoReferenceDemo {

    // see https://stanfordnlp.github.io/CoreNLP/coref.html
    public static void main(String[] args) {
        Annotation document = new Annotation("Barack Obama was born in Hawaii.  He is the president. Obama was elected in 2008.");

        // NOTE - all of these other models are required to do co-ref resolution
        Properties props = new Properties();
            props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,depparse,coref");

        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
            pipeline.annotate(document);
            System.out.println("---");
            System.out.println("coref chains");
            for (CorefChain cc : document.get(CorefCoreAnnotations.CorefChainAnnotation.class).values()) {
                System.out.println("\t" + cc);
            }

            for (CoreMap sentence : document.get(CoreAnnotations.SentencesAnnotation.class)) {
            System.out.println("---");
            System.out.println("mentions");
            for (Mention m : sentence.get(CorefCoreAnnotations.CorefMentionsAnnotation.class)) {
                System.out.println("\t" + m);
            }
        }
    }
}
