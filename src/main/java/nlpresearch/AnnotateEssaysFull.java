package nlpresearch;

import edu.stanford.nlp.coref.CorefCoreAnnotations;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.*;

public class AnnotateEssaysFull {

    // annotates all key tags in a sentence, including co-references
    public static final String DELIM = "->";
    public static final String DELIM_TAG = "|||";

    public static class Tuple<X, Y> {
        public final X x;
        public final Y y;
        public Tuple(X x, Y y) {
            this.x = x;
            this.y = y;
        }
    }

    private static List<String> findFiles(String folder){

        List<String> results = new ArrayList<String>();
        File[] files = new File(folder).listFiles();
        //If this pathname does not denote a directory, then listFiles() returns null.

        for (File file : files) {
            if (file.isFile() && file.getName().endsWith(".txt")) {
                results.add(file.getAbsolutePath());
            }
        }
        return results;
    }

    private static String readFile(String fname) throws IOException {
        StringBuilder sb = new StringBuilder();
        BufferedReader br = new BufferedReader(new FileReader(fname));
        try {

            String line = br.readLine();

            while (line != null) {
                sb.append(line);
                sb.append(System.lineSeparator());
                line = br.readLine();
            }
            String everything = sb.toString();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            br.close();
        }
        return sb.toString();
    }

    private static void addCoRefTags(Annotation document, List<Sentence> taggedSentences) {

        Integer headPhraseId = 0;
        for (CorefChain cc : document.get(CorefCoreAnnotations.CorefChainAnnotation.class).values()) {

            CorefChain.CorefMention representativeMention = cc.getRepresentativeMention();
            String headPhrase = representativeMention.mentionSpan.toLowerCase();
            headPhraseId++;

            tagWordsInMention(taggedSentences, representativeMention, COREF_ID, headPhraseId.toString());

            for (CorefChain.CorefMention mention : cc.getMentionsInTextualOrder()) {

                String mentionPhrase = mention.mentionSpan.toLowerCase();
                // a lot of mentions are the same text
                if(mentionPhrase.equals(headPhrase)){
                    continue;
                }

                tagWordsInMention(taggedSentences, mention, COREF_REF, headPhraseId.toString());
                // ensure is a single tag
                tagWordsInMention(taggedSentences, mention, COREF_PHRASE, headPhrase.replace(" ", "_"));
            }
        }
    }

    private static void tagWordsInMention(List<Sentence> taggedSentences, CorefChain.CorefMention mention, String tagName, String tagValue) {
        // note that all indexes and numbers are 1 based
        int sentNo = mention.sentNum - 1;
        int startIndex = mention.startIndex - 1;
        int endIndex = mention.endIndex - 2;

        Sentence sentence = taggedSentences.get(sentNo);

        for (int ix = startIndex; ix <= endIndex; ix++) {
            Word wd = sentence.words.get(ix);
            wd.AddTag(tagName, tagValue);
        }
    }

    public static void main(String[] args) throws IOException {

        final String DATASET = "CoralBleaching";
        final String PARTITION = "Test";

        String folder = "/Users/simon.hughes/Google Drive/Phd/Data/" + DATASET + "/Thesis_Dataset/CoReference/" + PARTITION;
        List<String> filenames = findFiles(folder);

        // see https://stanfordnlp.github.io/CoreNLP/coref.html
        // NOTE - all of these other models are required to do co-ref resolution
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,depparse,coref");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        int fileNo = 0;
        Instant start = Instant.now();
        Instant globalStart = Instant.now();

        for(String fname: filenames){

            if(fileNo % 10 == 0){
                System.out.print(new Integer(fileNo));
                Instant end = Instant.now();
                Duration dur = Duration.between(start, end);
                System.out.print(" : ");
                System.out.print(dur.getSeconds()/10);
                System.out.println(" secs per doc");
                start = Instant.now();
            }

            String text = readFile(fname);
            Annotation document = new Annotation( text);
            pipeline.annotate(document);

            List<Sentence> taggedSentences = getSentences(document);
            addCoRefTags(document, taggedSentences);

            List<String> lines = new ArrayList<>();

            for(Sentence sentence: taggedSentences){
                StringBuilder sbLine = new StringBuilder();
                for(Word wd: sentence.words){
                    sbLine.append(wd.token.toLowerCase()).append(DELIM);
                    for(Map.Entry entry: wd.tags.entrySet()){
                        sbLine.append(entry.getKey()).append(":").append(entry.getValue());
                        sbLine.append(DELIM_TAG);
                    }
                    sbLine.append(" ");
                }
                String line = sbLine.toString();
                // remove trailing tags
                line = line.replace(DELIM_TAG + " ", " ");
                lines.add(line.trim());
            }

            String outputFileName = fname + ".tagged";
            Path file = Paths.get(outputFileName);
            Files.write(file, lines, Charset.forName("UTF-8"));

            fileNo++;
        }
        System.out.println();
        System.out.println("Done");
        Instant globalEnd = Instant.now();
        Duration dur = Duration.between(globalStart, globalEnd);
        System.out.print("Took: ");
        System.out.println(dur);
    }

    private static final String POS  = "POS";
    private static final String NER  = "NER";
    private static final String COREF_ID  = "COREF_ID";
    private static final String COREF_REF  = "COREF_REF";
    private static final String COREF_PHRASE  = "COREF_PHRASE";

    private static List<Sentence> getSentences(Annotation document) {
        List<Sentence> sents = new ArrayList<>();

        List<CoreMap> coreSentences = document.get(CoreAnnotations.SentencesAnnotation.class);
        for(CoreMap sent: coreSentences){

            Sentence sentence = new Sentence();
            sents.add(sentence);

            for (CoreLabel token: sent.get(CoreAnnotations.TokensAnnotation.class)) {
                // this is the text of the token
                String tokenValue = token.get(CoreAnnotations.TextAnnotation.class);
                String pos = token.get(CoreAnnotations.PartOfSpeechAnnotation.class);
                if(pos.equals(":")){
                    pos = "COLON";
                }
                // this is the NER label of the token
                String ne = token.get(CoreAnnotations.NamedEntityTagAnnotation.class);
                Word wd = new Word(tokenValue);
                wd.AddTag(POS, pos);
                wd.AddTag(NER, ne);
                sentence.addWord(wd);
            }

        }
        return sents;
    }

    private static class Word{
        public String token;
        public Map<String,String> tags;

        public Word(String token){
            this.token = token;
            this.tags = new HashMap<>();
        }

        public void AddTag(String tagName, String value){
            this.tags.put(tagName, value);
        }
    }

    private static class Sentence{
        public List<AnnotateEssaysFull.Word> words = new ArrayList<AnnotateEssaysFull.Word>();

        public void addWord(AnnotateEssaysFull.Word word){
            this.words.add(word);
        }
    }

    //TODO: If the noun phrase in both is the same, then don't replace the text. If the head phrase is quite long, replace it just with the noun phrase (if there is one)
}
