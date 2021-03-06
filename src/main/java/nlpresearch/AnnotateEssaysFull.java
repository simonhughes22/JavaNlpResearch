package nlpresearch;

import edu.stanford.nlp.coref.CorefCoreAnnotations;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

import java.io.*;
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

    private static final String POS  = "POS";
    private static final String NER  = "NER";
    private static final String COREF_ID  = "COREF_ID";
    private static final String COREF_REF  = "COREF_REF";
    private static final String COREF_PHRASE  = "COREF_PHRASE";

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
                tagWordsInMention(taggedSentences, mention, COREF_PHRASE, headPhrase
                        .replace(" ", "_")
                        .replace(":", "_SEMI-COLON_"));
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

    private static void writeToFile(String fname, List<Sentence> taggedSentences, String extension) throws IOException {
        List<String> lines = new ArrayList<>();

        for(Sentence sentence: taggedSentences){
            StringBuilder sbLine = new StringBuilder();
            for(Word wd: sentence.words){
                sbLine.append(wd.token.toLowerCase()).append(DELIM);
                for(Map.Entry<String, Set<String>> entry: wd.tags.entrySet()){

                    Set<String> values = entry.getValue();
                    for(String val : values){
                        sbLine.append(entry.getKey()).append(":").append(val);
                        sbLine.append(DELIM_TAG);
                    }
                }
                sbLine.append(" ");
            }
            String line = sbLine.toString();
            // remove trailing tags
            line = line.replace(DELIM_TAG + " ", " ");
            lines.add(line.trim());
        }

        String outputFileName = fname + "." + extension;
        Path file = Paths.get(outputFileName);
        Files.write(file, lines, Charset.forName("UTF-8"));
    }

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
        public Map<String,Set<String>> tags;

        public Word(String token){
            this.token = token;
            this.tags = new HashMap<>();
        }

        public void AddTag(String tagName, String value){
            if(!this.tags.containsKey(tagName)){
                this.tags.put(tagName, new HashSet<String>());
            }
            this.tags.get(tagName).add(value);
        }
    }

    private static class Sentence{
        public List<AnnotateEssaysFull.Word> words = new ArrayList<AnnotateEssaysFull.Word>();

        public void addWord(AnnotateEssaysFull.Word word){
            this.words.add(word);
        }
    }

    private static Properties loadProperties(String fileName) {
        // see files here: https://github.com/stanfordnlp/CoreNLP/tree/master/src/edu/stanford/nlp/coref/properties

        InputStream input = AnnotateEssaysFull.class.getClass().getResourceAsStream("/" + fileName);
        Properties props = new Properties();
        try {
            props.load(input);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return props;
    }

    private static Properties createProperties(){
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse,coref");
        props.setProperty("coref.algorithm", "neural");
        return props;
    }

    private static void annotateDatasetPartition(String folder, StanfordCoreNLP pipeline) throws IOException {

        List<String> filenames = findFiles(folder);

        int fileNo = 0;
        Instant start = Instant.now();
        Instant globalStart = Instant.now();

        System.out.print("\t\t ");
        System.out.println(Integer.valueOf(filenames.size()).toString() + " files found");
        for(String fname: filenames){

            if(fileNo % 10 == 0 && fileNo > 0){
                System.out.print("\t\t ");
                System.out.print(Integer.valueOf(fileNo));
                Instant end = Instant.now();
                Duration dur = Duration.between(start, end);
                System.out.print(" : ");
                System.out.print(dur.getSeconds()/10);
                System.out.print(" secs per doc");
                System.out.println();
                start = Instant.now();
            }

            String text = readFile(fname);
            Annotation document = new Annotation( text);
            pipeline.annotate(document);

            List<Sentence> taggedSentences = getSentences(document);
            addCoRefTags(document, taggedSentences);

            writeToFile(fname, taggedSentences, "tagged");

            fileNo++;
        }
        System.out.println();
        System.out.println("Done");
        Instant globalEnd = Instant.now();
        Duration dur = Duration.between(globalStart, globalEnd);
        System.out.print("Took: ");
        System.out.println(dur);
    }

    public static void main(String[] args) throws IOException {

        // see https://stanfordnlp.github.io/CoreNLP/coref.html
        Properties props = loadProperties("neural-english.properties");
        List<String> datasets   = Arrays.asList("CoralBleaching", "SkinCancer");
        List<String> partitions = Arrays.asList("Test","Training");

        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        System.out.println("Stanford Core NLP library loaded\n");

        for(String dataset: datasets){
            for(String partition: partitions){
                String folder = "/Users/simon.hughes/Google Drive/Phd/Data/" + dataset + "/Thesis_Dataset/CoReference/" + partition;
                System.out.println("Dataset: " + dataset);
                System.out.println("\t Partition: " + partition);
                annotateDatasetPartition(folder, pipeline);
                System.out.println();
            }
        }
    }

    //TODO: If the noun phrase in both is the same, then don't replace the text. If the head phrase is quite long, replace it just with the noun phrase (if there is one)
}
