#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>
#include <set>
using namespace std;
using namespace cnn;
namespace po = boost::program_options;

struct Sentence {
  vector<int> ch;
  vector<int> lb;
  vector<int> rb;
  vector<int> pos;
  vector<int> tag;
  int len = 0;
  void addInstance(int ch_, int lb_, int rb_, int pos_, int tag_) {
    ch.push_back(ch_);
    lb.push_back(lb_);
    rb.push_back(rb_);
    pos.push_back(pos_);
    tag.push_back(tag_);
    len += 1;
  }
};

double pdrop = 0.5;
int LAYERS = 1;
unsigned HIDDEN_DIM = 128;
unsigned TAG_HIDDEN_DIM = 32;
unsigned TAG_SIZE = 0;
unsigned INPUT_DIM = 0;

unsigned CHAR_EMBEDDING_DIM = 0;
unsigned POS_EMBEDDING_DIM = 0;
unsigned WORD_EMBEDDING_DIM = 0;

int ALL_CHAR_SIZE = 0;
int ALL_POS_SIZE = 0;
int ALL_WORD_SIZE = 0;

bool eval = false;
bool usingPos = false;

vector<string> indexToTag;
unordered_map<string, int> wordToIndex;
unordered_map<string, int> charToIndex;
unordered_map<string, int> posToIndex;
unordered_map<string, int> tagToIndex;
unordered_map<int, vector<cnn::real>> wordEmbeddings;
unordered_map<int, vector<cnn::real>> charEmbeddings;

template <class Builder>
struct RNNLanguageModel {
  LookupParameters* p_w;
  LookupParameters* p_ch;
  LookupParameters* p_pos;
  Parameters* p_l2th;
  Parameters* p_r2th;
  Parameters* p_thbias;

  Parameters* p_th2t;
  Parameters* p_tbias;

  Parameters* p_w1;
  Parameters* p_w2;
  Parameters* p_w3;
  Parameters* p_b1;
  Parameters* p_b2;
  Parameters* p_EOS;
  Parameters* p_SOS;

  Builder l2rbuilder;
  Builder r2lbuilder;

  RNNLanguageModel(Model &model) :
    l2rbuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
    r2lbuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model) {
    p_ch = model.add_lookup_parameters(ALL_CHAR_SIZE, { CHAR_EMBEDDING_DIM });
    p_pos = model.add_lookup_parameters(ALL_POS_SIZE, { POS_EMBEDDING_DIM });
    p_w = model.add_lookup_parameters(ALL_WORD_SIZE, { WORD_EMBEDDING_DIM });
    for (auto x : wordEmbeddings) {
      p_w->Initialize(x.first, x.second);
    }

    p_l2th = model.add_parameters({ TAG_HIDDEN_DIM, HIDDEN_DIM });
    p_r2th = model.add_parameters({ TAG_HIDDEN_DIM, HIDDEN_DIM });
    p_thbias = model.add_parameters({ TAG_HIDDEN_DIM });
    p_th2t = model.add_parameters({ TAG_SIZE, TAG_HIDDEN_DIM });
    p_tbias = model.add_parameters({ TAG_SIZE });

    p_SOS = model.add_parameters({ INPUT_DIM });
    p_EOS = model.add_parameters({ INPUT_DIM });
    p_w1 = model.add_parameters({ WORD_EMBEDDING_DIM, WORD_EMBEDDING_DIM });
    p_w2 = model.add_parameters({ WORD_EMBEDDING_DIM, WORD_EMBEDDING_DIM });
    p_b1 = model.add_parameters({ WORD_EMBEDDING_DIM });

    p_w3 = model.add_parameters({ INPUT_DIM, CHAR_EMBEDDING_DIM + WORD_EMBEDDING_DIM * 2 + POS_EMBEDDING_DIM });
    p_b2 = model.add_parameters({ INPUT_DIM });
  }
  vector<cnn::real> getVec(int x, int len) {
    vector<cnn::real> ret(len, 0);
    ret[x] = 1.0;
    return ret;
  }
  Expression BuildTaggingGraph(const Sentence &s, ComputationGraph &cg,
    double &cor, double &nTagged, vector<int> *pt = nullptr) {
    const int slen = s.len;
    l2rbuilder.new_graph(cg);
    l2rbuilder.start_new_sequence();
    r2lbuilder.new_graph(cg);
    r2lbuilder.start_new_sequence();

    Expression i_l2th = parameter(cg, p_l2th);
    Expression i_r2th = parameter(cg, p_r2th);
    Expression i_thbias = parameter(cg, p_thbias);
    Expression i_th2t = parameter(cg, p_th2t);
    Expression i_tbias = parameter(cg, p_tbias);


    Expression i_w1 = parameter(cg, p_w1);
    Expression i_w2 = parameter(cg, p_w2);
    Expression i_w3 = parameter(cg, p_w3);
    Expression i_b1 = parameter(cg, p_b1);
    Expression i_b2 = parameter(cg, p_b2);


    vector<Expression> errs(slen);
    vector<Expression> i_words(slen);
    vector<Expression> fwds(slen);
    vector<Expression> revs(slen);

    l2rbuilder.add_input(parameter(cg, p_SOS));

    for (int i = 0; i < slen; i++) {
      auto it_ch = charEmbeddings.find(s.ch[i]);
      Expression charInput;
      Expression wordInput;
      Expression posInput;

      charInput = lookup(cg, p_ch, s.ch[i]);
      wordInput = concatenate({ lookup(cg, p_w, s.lb[i]), lookup(cg, p_w, s.rb[i]) });
      //rectify(affine_transform({ i_b1, i_w1, lookup(cg, p_w, s.lb[i]), i_w2, lookup(cg, p_w, s.rb[i]) }));
      posInput = lookup(cg, p_pos, s.pos[i]);

      i_words[i] = concatenate({ charInput, wordInput, posInput });

      i_words[i] = rectify(affine_transform({ i_b2, i_w3, i_words[i] }));

      //if (!eval) { iWords.set(t, Expression.Creator.noise(iWords.get(t), 0.1)); }
      fwds[i] = l2rbuilder.add_input(i_words[i]);
    }

    r2lbuilder.add_input(parameter(cg, p_EOS));
    for (int i = 0; i < slen; i++)
      revs[slen - i - 1] = r2lbuilder.add_input(i_words[slen - i - 1]);

    for (int i = 0; i < slen; i++) {
      nTagged++;
      Expression i_th = rectify(affine_transform({ i_thbias, i_l2th, fwds[i], i_r2th, revs[i] }));
      //if (!eval) { iTh = Expression.Creator.dropout(iTh, pDrop); }
      Expression i_t = affine_transform({ i_tbias, i_th2t, i_th });
      vector<cnn::real> dist = as_vector(cg.incremental_forward());
      double best = -1e100;
      int bestj = -1;
      for (int j = 0; j < dist.size(); j++) {
        if (dist[j] > best) {
          best = dist[j];
          bestj = j;
        }
      }
      if (pt != nullptr) pt->push_back(bestj);
      if (s.tag[i] == bestj) cor++;
      Expression i_err = pickneglogsoftmax(i_t, s.tag[i]);
      errs[i] = i_err;
      /*if (indexToTag[s.tag[i]][0] == 'B' && indexToTag[bestj][0] == 'O') {
      errs[i] = errs[i] * 5;
      }*/
    }
    return sum(errs);
  }
};

void init_command_line(int argc, char* argv[], po::variables_map* conf) {
  po::options_description opts("LSTM-CGED");
  opts.add_options()
    ("training_data", po::value<std::string>(), "The path to the training data.")
    ("devel_data", po::value<std::string>(), "The path to the development data.")
    ("testing_data", po::value<std::string>(), "The path to the testing data.")
    ("pretrained", po::value<std::string>()->default_value(""), "The path to the pretrained bigram word embedding.")
    ("model_name", po::value<std::string>(), "The path to the model of this training phase.")
    ("model_file", po::value<string>()->default_value(""), "The path to the model which was trained before.")
    ("layers", po::value<int>()->default_value(1), "number of LSTM layers")
    ("bigram_dim", po::value<int>()->default_value(100), "input bigram embedding size")
    ("unigram_dim", po::value<int>()->default_value(50), "unigram dimension")
    ("pos_dim", po::value<int>()->default_value(16), "POS dimension")
    ("hidden_dim", po::value<int>()->default_value(128), "hidden dimension")
    ("input_dim", po::value<int>()->default_value(128), "LSTM input dimension")
    ("maxiter", po::value<int>()->default_value(10), "Max number of iterations.")
    ("log_file", po::value<string>()->default_value(""), "The path to the predicted file.")
    ("help,h", "Show help information");

  po::store(po::parse_command_line(argc, argv, opts), *conf);
  if (conf->count("help")) {
    std::cerr << opts << std::endl;
    exit(1);
  }
}

vector<string> split(string str, char ch) {
  vector<string> ret;
  string s = "";
  for (int i = 0; i <= str.size(); i++) {
    if (i == str.size() || str[i] == ch) {
      ret.push_back(s);
      s = "";
    }
    else {
      s += str[i];
    }
  }
  return ret;
}

void readFile(string fileName, vector<Sentence> &data) {
  ifstream in(fileName);
  int lc = 0, toks = 0;
  string sentence;
  while (getline(in, sentence)) {
    if (lc % 1000 == 0) cerr << lc << endl;
    vector<string> words = split(sentence, ' ');
    vector<string> temp;
    temp.push_back("<SOS>");
    for (int i = 0; i < words.size(); ++i) {
      vector<string> item = split(words[i], '&');
      temp.push_back(item[0]);
    }
    temp.push_back("<EOS>");
    data.push_back(Sentence());
    for (int i = 0; i < words.size(); ++i) {
      vector<string> item = split(words[i], '&');
      string lb = temp[i] + temp[i + 1];
      string rb = temp[i + 1] + temp[i + 2];

      if (wordToIndex.find(lb) == wordToIndex.end()) {
        wordToIndex.insert(make_pair(lb, wordToIndex.size()));
      }
      if (wordToIndex.find(rb) == wordToIndex.end()) {
        wordToIndex.insert(make_pair(rb, wordToIndex.size()));
      }
      if (charToIndex.find(item[0]) == charToIndex.end()) {
        charToIndex.insert(make_pair(item[0], charToIndex.size()));
      }
      if (posToIndex.find(item[1]) == posToIndex.end()) {
        posToIndex.insert(make_pair(item[1], posToIndex.size()));
      }
      if (tagToIndex.find(item[2]) == tagToIndex.end()) {
        indexToTag.push_back(item[2]);
        tagToIndex.insert(make_pair(item[2], tagToIndex.size()));
      }
      data[lc].addInstance(charToIndex[item[0]], wordToIndex[lb], wordToIndex[rb], posToIndex[item[1]], tagToIndex[item[2]]);
    }
    lc++;
    toks += words.size();
  }
  cerr << lc << " lines, " << toks << " tokens, " << endl;
  cerr << "word, pos tag:" << wordToIndex.size() << " " << posToIndex.size() << " " << tagToIndex.size() << endl;
}

void readEmbedding(string fileName) {
  ifstream in(fileName);
  string line;
  getline(in, line);
  int dim = atoi(split(line, ' ')[1].c_str());
  int cnt = 0;
  while (getline(in, line)) {
    vector<string> item = split(line, ' ');
    cnt++;
    string word = item[0];
    vector<cnn::real> e(dim);
    for (int i = 1; i <= dim; ++i)
      e[i - 1] = atof(item[i].c_str());
    if (wordToIndex.find(word) != wordToIndex.end() && wordEmbeddings.find(wordToIndex[word]) == wordEmbeddings.end())
      wordEmbeddings[wordToIndex[word]] = e;
    if (cnt % 100000 == 0) cerr << cnt << endl;
  }
}

int getCurTime() {
  return clock();
}

void calculateNer(const Sentence s, vector<int> pt, double &tp, double &fp, double &fn) {
  set <int> seg;
  for (int i = 0; i < s.len; ++i) {
    string cur = indexToTag[s.tag[i]];
    if (cur[0] == 'B') {
      int j = i + 1;
      while (j < s.len) {
        string str = indexToTag[s.tag[j]];
        if (!(str[0] == 'I' && cur.substr(1) == str.substr(1)))
          break;
        ++j;
      }
      seg.insert(100000000 * s.tag[i] + 10000 * i + j - 1);
      i = j - 1;
    }
  }
  double tp_ = 0.0;
  double fp_ = 0.0;
  double fn_ = 0.0;
  int cnt = 0;
  for (int i = 0; i < pt.size(); ++i) {
    string cur = indexToTag[pt[i]];
    if (cur[0] == 'B') {
      int j = i + 1;
      while (j < pt.size()) {
        string str = indexToTag[pt[j]];
        if (!(str[0] == 'I' && cur.substr(1) == str.substr(1)))
          break;
        j++;
      }
      cnt++;
      if (seg.find(100000000 * pt[i] + i * 10000 + j - 1) != seg.end()) tp_++;
      else fp_++;
      i = j - 1;
    }
  }
  fn_ = seg.size() - tp_;
  tp += tp_;
  fp += fp_;
  fn += fn_;
}

double bestF = 0.0;

void runOnDev(RNNLanguageModel<LSTMBuilder> &lm, const Model &model, const vector<Sentence> dev, const string &modelName) {
  int last = getCurTime();
  double dloss = 0.0;
  double dcorr = 0.0;
  double dtags = 0.0;
  double tp = 0.0;
  double fp = 0.0;
  double fn = 0.0;
  double P = 0.0;
  double R = 0.0;
  double F = 0.0;
  eval = true;
  for (int i = 0; i < dev.size(); ++i) {
    ComputationGraph cg;
    vector<int> pt;
    lm.BuildTaggingGraph(dev[i], cg, dcorr, dtags, &pt);
    //cout << i << endl;
    calculateNer(dev[i], pt, tp, fp, fn);
  }
  eval = false;
  P = tp / (tp + fp);
  R = tp / (tp + fn);
  F = 2 * P * R / (P + R);
  if (F > bestF) {
    bestF = F;
    ofstream out(modelName);
    boost::archive::text_oarchive oa(out);
    oa << model;
  }
  cerr << "\n***DEV" << " E = " << dloss / dtags << " ppl = "
    << exp(dloss / dtags) << " acc = " << dcorr / dtags << "[consume = " << (getCurTime() - last) / 1000.0 << "s]" << endl;
  cerr << "P = " << P << " R = " << R << " F = " << F << endl;
}

void doPredict(RNNLanguageModel<LSTMBuilder> &lm, const Model &model, const vector<Sentence> dev, const string &logName) {
  ofstream out(logName);
  double dloss = 0.0;
  double dcorr = 0.0;
  double dtags = 0.0;
  eval = true;
  for (int i = 0; i < dev.size(); ++i) {
    ComputationGraph cg;
    vector<int> pt;
    lm.BuildTaggingGraph(dev[i], cg, dcorr, dtags, &pt);
    for (int j = 0; j < pt.size(); ++j) {
      out << indexToTag[pt[j]] << endl;
    }
    out << endl;
  }
  eval = false;
}

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  po::variables_map conf;
  init_command_line(argc, argv, &conf);

  string trainName = conf["training_data"].as<string>();
  string devName = conf["devel_data"].as<string>();
  string testName = conf["testing_data"].as<string>();
  string embeddingName = conf["pretrained"].as<string>();
  string modelName = conf["model_name"].as<string>();
  INPUT_DIM = conf["input_dim"].as<int>();
  HIDDEN_DIM = conf["hidden_dim"].as<int>();
  LAYERS = conf["layers"].as<int>();
  CHAR_EMBEDDING_DIM = conf["unigram_dim"].as<int>();
  WORD_EMBEDDING_DIM = conf["bigram_dim"].as<int>();
  POS_EMBEDDING_DIM = conf["pos_dim"].as<int>();
  string logName = conf["log_file"].as<string>();

  vector<Sentence> train;
  vector<Sentence> dev;
  vector<Sentence> test;

  cout << "Reading training data from " << trainName << "..." << endl;

  readFile(trainName, train);

  cout << "Reading devel data from " << devName << "..." << endl;

  readFile(devName, dev);

  cout << "Reading testing data from " << testName << "..." << endl;

  readFile(testName, test);

  cout << "Reading embedding data from " << embeddingName << "..." << endl;
  readEmbedding(embeddingName);

  int maxIteration = 100;
  int numInstances = train.size(); //Math.min(2000, trainX.size());
  maxIteration = conf["maxiter"].as<int>();
  numInstances = numInstances;

  Model model;
  bool useMomentum = false;
  Trainer* sgd = nullptr;
  if (useMomentum)
    sgd = new MomentumSGDTrainer(&model);
  else
    sgd = new SimpleSGDTrainer(&model);


  ALL_CHAR_SIZE = charToIndex.size();
  ALL_POS_SIZE = posToIndex.size();
  ALL_WORD_SIZE = wordToIndex.size();


  TAG_SIZE = tagToIndex.size();
  if (usingPos) INPUT_DIM += posToIndex.size();

  RNNLanguageModel<LSTMBuilder> lm(model);
  if (conf["model_file"].as<string>() != "") {
    ifstream in(conf["model_file"].as<string>());
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }
  vector<int> order;
  for (int i = 0; i < train.size(); i++)
    order.push_back(i);

  for (int i = 0; i < indexToTag.size(); i++)
    cout << indexToTag[i] << endl;
  int last = getCurTime();
  int tot = last;
  for (int iteration = 0; iteration < maxIteration; ++iteration) {
    double loss = 0.0;
    double correct = 0.0;
    double ttags = 0.0;
    random_shuffle(order.begin(), order.end());

    for (int i = 0; i < order.size(); i++) {
      {
        int index = order[i];
        ComputationGraph cg;
        lm.BuildTaggingGraph(train[index], cg, correct, ttags);
        loss += as_scalar(cg.incremental_forward());
        cg.backward();
        sgd->update(1.0);
      }
      if (i + iteration > 0 && i % 5000 == 0) {
        runOnDev(lm, model, dev, modelName);
      }
      if (i + iteration > 0 && i % 50 == 0) {
        cerr << "E = " << (loss / ttags) << " ppl = " << exp(loss / ttags)
          << " (acc = " << (correct / ttags) << ")" << " iterations : " << iteration
          << " lines : " << i << "[consume = " << (getCurTime() - last) / 1000.0 << "s]" << endl;;
        last = getCurTime();
      }
    }
    sgd->update_epoch();
    cerr << "Iteration Time : " << (getCurTime() - tot) / 1000.0 << "s]" << endl;
    tot = getCurTime();
  }

  ifstream in(modelName);
  boost::archive::text_iarchive ia(in);
  ia >> model;

  doPredict(lm, model, test, logName);
  delete sgd;
}

