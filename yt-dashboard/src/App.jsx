import { useState, useEffect } from 'react';
import { 
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell
} from 'recharts';
import { 
  LayoutDashboard, TrendingUp, Search, Clock, Hash, Video, Globe, List, ExternalLink, Zap, Info
} from 'lucide-react';

function App() {
  const [summaryData, setSummaryData] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [searchVideo, setSearchVideo] = useState('');
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [timeSeriesData, setTimeSeriesData] = useState(null);
  const [predictionCases, setPredictionCases] = useState([]);
  const [predictionUrl, setPredictionUrl] = useState('');
  const [watchlist, setWatchlist] = useState(['UC2rxmP4xIG8aPArxvx5fXxQ', 'UCDpeY-XhvHTW-vW2BfB__mw']);
  const [watchlistHistory, setWatchlistHistory] = useState({});
  const [newChannel, setNewChannel] = useState('');
  const [predHubTab, setPredHubTab] = useState('experiments'); 
  const [predLabTab, setPredLabTab] = useState('channel');
  const [quickPredResult, setQuickPredResult] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  
  const [liveTitle, setLiveTitle] = useState('');
  const [liveViews, setLiveViews] = useState('');
  const [livePubDate, setLivePubDate] = useState(new Date().toISOString().slice(0, 16));
  const [showManualForm, setShowManualForm] = useState(false);
  const [metadataError, setMetadataError] = useState(null);
  
  // States for Live Predictor Inputs
  const [liveThumbnail, setLiveThumbnail] = useState('');
  const [liveDuration, setLiveDuration] = useState(0);
  const [liveChannelID, setLiveChannelID] = useState('');
  const [liveChannelAvg, setLiveChannelAvg] = useState(0);

  // Poll Watchlist on Load
  useEffect(() => {
    watchlist.forEach(id => syncChannel(id));
  }, []);

  const syncChannel = (chanId) => {
    fetch(`http://localhost:8000/fetch_channel_latest/${chanId}`)
      .then(res => res.json())
      .then(data => {
        setWatchlistHistory(prev => ({ ...prev, [chanId]: data }));
      })
      .catch(err => console.error("Watchlist sync error:", err));
  };

  const fetchRealMetadata = (vidId) => {
    setIsPredicting(true);
    setMetadataError(null);
    setQuickPredResult(null);
    setShowManualForm(false);
    
    fetch(`http://localhost:8000/fetch_metadata/${vidId}`)
      .then(res => {
        if(!res.ok) throw new Error("Video không tồn tại hoặc lỗi API");
        return res.json();
      })
      .then(data => {
        setLiveTitle(data.title);
        setLiveViews(data.current_views);
        setLivePubDate(data.published_at.replace(' ', 'T').substring(0, 16));
        setLiveThumbnail(data.thumbnail);
        setLiveDuration(data.duration_sec);
        setLiveChannelID(data.channel_id);
        setLiveChannelAvg(data.channel_avg_views);
        setShowManualForm(true);
        setIsPredicting(false);
        
        // Auto-Run Prediction
        runLiveInference(data.title, data.current_views, data.published_at, data.duration_sec, data.channel_avg_views);
      })
      .catch(err => {
        console.error(err);
        setIsPredicting(false);
        setMetadataError(err.message);
      });
  };

  const runLiveInference = (title, views, pubDate, duration, chanAvg) => {
    setIsPredicting(true);
    fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: title,
        current_views: parseFloat(views),
        published_at: pubDate,
        duration_sec: duration || 60,
        channel_avg_views: chanAvg || 1000
      })
    })
    .then(res => res.json())
    .then(data => {
      setQuickPredResult({
        title: data.title,
        views24: data.prediction_24h,
        views48: data.prediction_48h,
        viralStatus: data.viral_status,
        age: data.age_hours,
        model: data.model
      });
      setIsPredicting(false);
    })
    .catch(err => {
      console.error(err);
      setIsPredicting(false);
    });
  };

  useEffect(() => {
    fetch('/data/summary.json')
      .then(res => res.json())
      .then(data => setSummaryData(data))
      .catch(err => console.error("Error fetching summary data:", err));

    fetch('/data/prediction_cases.json')
      .then(res => res.json())
      .then(data => setPredictionCases(data))
      .catch(err => console.error("Error fetching prediction cases:", err));

    fetch('/data/watchlist_history.json')
      .then(res => res.json())
      .then(data => setWatchlistHistory(data))
      .catch(err => console.error("Error fetching watchlist history:", err));
  }, []);

  const loadTimeSeries = (video) => {
    setSelectedVideo(video);
    fetch(`/data/timeseries/${video.video_id}.json`)
      .then(res => res.json())
      .then(data => {
        // Format history timestamps
        const formatted = data.history.map(pt => ({
          ...pt,
          timeLabel: new Date(pt.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
        }));
        setTimeSeriesData(formatted);
      })
      .catch(err => {
        console.error("Error fetching timeseries:", err);
        setTimeSeriesData([]);
      });
  };

  if (!summaryData) {
    return (
      <div className="app-container" style={{justifyContent: 'center', alignItems: 'center'}}>
        <h2 className="animate-fade-in">Loading Analytics...</h2>
      </div>
    );
  }

  // Calculate posting time effectiveness (views by hour of day published)
  const postingHoursData = Array.from({length: 24}, (_, i) => ({ hour: i, views: 0, count: 0 }));
  summaryData.top_videos.forEach(vid => {
    if (vid.published_at) {
      const h = new Date(vid.published_at).getUTCHours();
      postingHoursData[h].views += vid.views;
      postingHoursData[h].count += 1;
    }
  });

  const avgViewsByHour = postingHoursData.map(d => ({
    hour: `${d.hour}:00`,
    avgViews: d.count > 0 ? Math.round(d.views / d.count) : 0
  })).filter(d => d.avgViews > 0);

  const topVideoRows = summaryData.top_videos
    .filter(vid => vid.title.toLowerCase().includes(searchVideo.toLowerCase()) || vid.video_id.includes(searchVideo))
    .slice(0, 10);

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar animate-fade-in">
        <div className="sidebar-logo">
          <Video size={32} color="var(--primary)" />
          ShortsDash
        </div>
        
        <div 
          className={`nav-link ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          <LayoutDashboard size={20} /> Overview
        </div>
        <div 
          className={`nav-link ${activeTab === 'rankings' ? 'active' : ''}`}
          onClick={() => setActiveTab('rankings')}
        >
          <TrendingUp size={20} /> Top Rankings
        </div>
        <div 
          className={`nav-link ${activeTab === 'categories' ? 'active' : ''}`}
          onClick={() => setActiveTab('categories')}
        >
          <Hash size={20} /> Hashtags / Categories
        </div>
        <div 
          className={`nav-link ${activeTab === 'analytics' ? 'active' : ''}`}
          onClick={() => setActiveTab('analytics')}
        >
          <Globe size={20} /> Advanced Analytics
        </div>
        <div 
          className={`nav-link ${activeTab === 'prediction' ? 'active' : ''}`}
          onClick={() => setActiveTab('prediction')}
        >
          <TrendingUp size={20} /> Prediction Hub
        </div>
        <div 
          className={`nav-link ${activeTab === 'growth' ? 'active' : ''}`}
          onClick={() => setActiveTab('growth')}
        >
          <Search size={20} /> Growth Tracker
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <header style={{marginBottom: '2rem'}}>
          <h1 className="animate-fade-in">
            {activeTab === 'overview' && 'Dashboard Overview'}
            {activeTab === 'rankings' && 'Video Rankings'}
            {activeTab === 'categories' && 'Hashtags Matrix'}
            {activeTab === 'analytics' && 'Country & Category Analytics'}
            {activeTab === 'prediction' && 'Real-Time Prediction Hub'}
            {activeTab === 'growth' && 'Growth Tracking & Analysis'}
          </h1>
          <p style={{color: 'var(--text-muted)'}} className="animate-fade-in">
            Analyzing {summaryData.total_videos_analyzed.toLocaleString()} YouTube Shorts • {summaryData.top_videos.length} Top Performers Enriched
          </p>
        </header>

        {/* OVERVIEW TAB */}
        {activeTab === 'overview' && (
          <div className="animate-fade-in">
            <div className="grid-3">
              <div className="card">
                <div className="card-title"><Video size={20} /> Total Tracked</div>
                <div className="stat-value">{summaryData.total_videos_analyzed.toLocaleString()}</div>
                <div className="stat-label">Videos in Database</div>
              </div>
              <div className="card">
                <div className="card-title"><TrendingUp size={20} /> Highest Views</div>
                <div className="stat-value">{summaryData.top_videos[0].views.toLocaleString()}</div>
                <div className="stat-label">Top Video: {summaryData.top_videos[0].title.substring(0, 30)}...</div>
              </div>
              <div className="card">
                <div className="card-title"><Hash size={20} /> Top Category</div>
                <div className="stat-value">#{summaryData.top_hashtags[0].tag}</div>
                <div className="stat-label">{summaryData.top_hashtags[0].views.toLocaleString()} Total Views</div>
              </div>
            </div>

            <div className="grid-2">
              <div className="card">
                <div className="card-title"><Globe size={20} /> Geographic Distribution (Top)</div>
                <div style={{width: '100%', height: 300}}>
                  <ResponsiveContainer>
                    <PieChart>
                      <Pie
                        data={summaryData.country_analytics.slice(0, 7)}
                        cx="50%" cy="50%" innerRadius={60} outerRadius={80} paddingAngle={5} dataKey="views"
                      >
                        {summaryData.country_analytics.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={['#9b87f5', '#7E69AB', '#D6BCFA', '#F1F0FB', '#8E9196'][index % 5]} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div className="card">
                <div className="card-title"><Clock size={20} /> Effective Posting Time Analysis (UTC)</div>
                <div style={{width: '100%', height: 300}}>
                  <ResponsiveContainer>
                    <BarChart data={avgViewsByHour}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="hour" stroke="var(--text-muted)" />
                      <YAxis stroke="var(--text-muted)" />
                      <Tooltip 
                        contentStyle={{backgroundColor: 'var(--bg-dark)', borderColor: 'var(--border)'}}
                        itemStyle={{color: 'var(--primary)'}}
                      />
                      <Bar dataKey="avgViews" fill="var(--primary)" radius={[4,4,0,0]} name="Avg Views" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* RANKINGS TAB */}
        {activeTab === 'rankings' && (
          <div className="card animate-fade-in">
            <div className="card-title">Top Performing Shorts</div>
            <input 
              type="text" 
              className="search-input" 
              placeholder="Search by title, video ID, or channel..." 
              value={searchVideo}
              onChange={(e) => setSearchVideo(e.target.value)}
            />
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Short Preview</th>
                    <th>Statistics</th>
                    <th>Metadata</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {topVideoRows.map((vid, idx) => (
                    <tr key={vid.video_id}>
                      <td><span className="badge">#{summaryData.top_videos.findIndex(v => v.video_id === vid.video_id) + 1}</span></td>
                      <td style={{display: 'flex', gap: '1rem', alignItems: 'center'}}>
                        <img 
                          src={`https://i.ytimg.com/vi/${vid.video_id}/mqdefault.jpg`} 
                          alt="Thumbnail" 
                          style={{width: '120px', borderRadius: '8px', border: '1px solid var(--border)'}}
                        />
                        <div style={{maxWidth: '250px'}}>
                          <div className="text-trunc" title={vid.title} style={{fontWeight: 600, fontSize: '0.95rem'}}>{vid.title}</div>
                          <div style={{fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px'}}>
                            <a href={`https://youtube.com/shorts/${vid.video_id}`} target="_blank" rel="noreferrer" style={{display: 'flex', alignItems: 'center', gap: '4px'}}>
                              Watch on YouTube <ExternalLink size={12} />
                            </a>
                          </div>
                        </div>
                      </td>
                      <td>
                        <div style={{fontWeight: 'bold', fontSize: '1.1rem'}}>{vid.views.toLocaleString()} <span style={{fontSize: '0.7rem', color: 'var(--text-muted)'}}>views</span></div>
                        <div style={{fontSize: '0.8rem', color: 'var(--text-muted)'}}>{vid.likes.toLocaleString()} likes</div>
                      </td>
                      <td>
                        <div className="badge" style={{background: 'rgba(255,255,255,0.05)', color: 'var(--text-light)', border: '1px solid var(--border)', marginBottom: '4px'}}>
                          {vid.category}
                        </div>
                        <div style={{fontSize: '0.8rem', color: 'var(--primary)', fontWeight: 500}}>{vid.country}</div>
                      </td>
                      <td>
                        <button 
                          style={{
                            background: 'var(--primary)', color: 'white', border: 'none', 
                            padding: '8px 12px', borderRadius: '6px', cursor: 'pointer', transition: '0.2s'
                          }}
                          onClick={() => {
                            setActiveTab('growth');
                            loadTimeSeries(vid);
                          }}
                        >
                          Analysis
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ANALYTICS TAB */}
        {activeTab === 'analytics' && (
          <div className="animate-fade-in">
            <div className="grid-2">
              <div className="card">
                <div className="card-title"><List size={20} /> Categories by Popularity</div>
                <div style={{width: '100%', height: 400}}>
                  <ResponsiveContainer>
                    <BarChart data={summaryData.category_analytics.sort((a,b) => b.views - a.views)} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis type="number" stroke="var(--text-muted)" />
                      <YAxis dataKey="name" type="category" stroke="var(--text-muted)" width={100} />
                      <Tooltip />
                      <Bar dataKey="views" fill="var(--primary)" radius={[0,4,4,0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div className="card">
                <div className="card-title"><Globe size={20} /> Top Countries by Views</div>
                <div style={{width: '100%', height: 400}}>
                  <ResponsiveContainer>
                    <BarChart data={summaryData.country_analytics.sort((a,b) => b.views - a.views)} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis type="number" stroke="var(--text-muted)" />
                      <YAxis dataKey="name" type="category" stroke="var(--text-muted)" width={100} />
                      <Tooltip />
                      <Bar dataKey="views" fill="#D6BCFA" radius={[0,4,4,0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* CATEGORIES TAB */}
        {activeTab === 'categories' && (
          <div className="card animate-fade-in">
            <div className="card-title"><Hash size={20} /> Top Hashtags & Categories</div>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Hashtag</th>
                    <th>Frequency</th>
                    <th>Aggregated Views</th>
                  </tr>
                </thead>
                <tbody>
                  {summaryData.top_hashtags.map((ht, idx) => (
                    <tr key={ht.tag}>
                      <td><span className="badge">#{idx + 1}</span></td>
                      <td style={{fontWeight: 600, color: 'var(--primary)'}}>#{ht.tag}</td>
                      <td>{ht.count.toLocaleString()} videos</td>
                      <td style={{fontWeight: 'bold'}}>{ht.views.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* PREDICTION HUB TAB */}
        {activeTab === 'prediction' && (
          <div className="animate-fade-in">
            <div className="sub-tabs">
              <button className={`sub-tab ${predHubTab === 'experiments' ? 'active' : ''}`} onClick={() => setPredHubTab('experiments')}>
                Model Verification
              </button>
              <button className={`sub-tab ${predHubTab === 'watchlist' ? 'active' : ''}`} onClick={() => setPredHubTab('watchlist')}>
                Smart Watchlist
              </button>
              <button className={`sub-tab ${predHubTab === 'live' ? 'active' : ''}`} onClick={() => setPredHubTab('live')}>
                Lightning Predictor
              </button>
            </div>

            {/* TAB 1: EXPERIMENTS / VERIFICATION */}
            {predHubTab === 'experiments' && (
              <div className="card animate-fade-in">
                <div className="card-title"><TrendingUp size={20} /> Part 1: Historical Experiments (XGBoost)</div>
                <div className="viral-standard-label">
                   <Info size={16} /> <strong>Viral Standard:</strong> Total views đạt mốc 1.000.000+
                </div>
                
                <div className="sub-tabs" style={{background: 'rgba(255,255,255,0.03)'}}>
                  <button className={`sub-tab ${predLabTab === 'channel' ? 'active' : ''}`} onClick={() => setPredLabTab('channel')}>Generalization</button>
                  <button className={`sub-tab ${predLabTab === 'time' ? 'active' : ''}`} onClick={() => setPredLabTab('time')}>Trend Prediction</button>
                </div>

                <div className="table-container">
                  <table style={{fontSize: '0.85rem'}}>
                    <thead>
                      <tr>
                        <th style={{width: '100px'}}>Preview</th>
                        <th>Video Title</th>
                        <th>Views 24h (Act/Pred)</th>
                        <th>Viral Prediction</th>
                        <th>Result Status</th>
                        <th>Accuracy</th>
                      </tr>
                    </thead>
                    <tbody>
                      {predictionCases.filter(c => predLabTab === 'channel' ? c.split === 'Channel-Based' : c.split === 'Time-Based').map((c, i) => (
                        <tr key={i}>
                          <td><img src={c.thumbnail} className="table-row-thumb" alt="thumb" /></td>
                          <td style={{verticalAlign: 'top', paddingTop: '15px'}}>
                            <div className="text-trunc" style={{fontWeight: 700, fontSize: '0.95rem'}} title={c.title}>{c.title}</div>
                            <div style={{fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px'}}>ID: {c.video_id}</div>
                          </td>
                          <td>
                            <div style={{fontSize: '1rem', fontWeight: 600}}>{c.actual_24h.toLocaleString()}</div>
                            <div className="status-badge" style={{
                              marginTop: '8px',
                              background: c.actual_label === 'VIRAL' ? 'gold' : c.actual_label === 'POTENTIAL' ? 'rgba(74, 222, 128, 0.2)' : 'rgba(255,255,255,0.05)',
                              color: c.actual_label === 'VIRAL' ? 'black' : 'white'
                            }}>
                               {c.actual_label}
                            </div>
                          </td>
                          <td>
                            <div style={{fontSize: '1rem', fontWeight: 700, color: 'var(--primary)'}}>{c.pred_24h.toLocaleString()}</div>
                            <div className="status-badge" style={{
                              marginTop: '8px',
                              background: c.pred_label === 'VIRAL' ? 'gold' : c.pred_label === 'POTENTIAL' ? 'rgba(74, 222, 128, 0.2)' : 'rgba(255,255,255,0.05)',
                              color: c.pred_label === 'VIRAL' ? 'black' : 'white'
                            }}>
                               {c.pred_label}
                            </div>
                          </td>
                          <td>
                            <span className={`status-badge ${c.status === 'SUCCESS' ? 'status-success' : c.status === 'MISS' ? 'status-miss' : 'status-alarm'}`}>
                               {c.status}
                            </span>
                          </td>
                          <td>
                            <span style={{color: c.error_24h_pct < 20 ? '#4ade80' : '#f87171'}}>
                              {(100 - c.error_24h_pct).toFixed(1)}%
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* TAB 2: WATCHLIST */}
            {predHubTab === 'watchlist' && (
              <div className="card animate-fade-in">
                <div className="card-title"><Video size={20} /> Part 2: Smart Watchlist Monitoring</div>
                <div style={{display: 'flex', gap: '10px', marginBottom: '2rem'}}>
                  <input type="text" className="search-input" style={{marginBottom: 0}} placeholder="Enter Channel ID..." value={newChannel} onChange={(e) => setNewChannel(e.target.value)} />
                  <button className="sub-tab active" style={{padding: '0 20px'}} onClick={() => { if(newChannel) setWatchlist([...watchlist, newChannel]); setNewChannel(''); }}>Add Channel</button>
                </div>
                <div className="grid-2">
                  {watchlist.map(id => (
                    <div key={id} className="card" style={{border: '1px solid var(--border)', background: 'rgba(0,0,0,0.2)'}}>
                      <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px'}}>
                        <span style={{fontWeight: 700, color: 'var(--primary)'}}>{id}</span>
                        <span className="badge" style={{background: 'rgba(74, 222, 128, 0.1)', color: '#4ade80'}}>Live Scanning</span>
                      </div>
                      {watchlistHistory[id] ? (
                        <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '10px'}}>
                          {watchlistHistory[id].map(v => (
                            <div key={v.video_id} style={{fontSize: '0.75rem'}}>
                              <img src={v.thumbnail} style={{width: '100%', borderRadius: '6px', marginBottom: '5px'}} />
                              <div className="text-trunc" style={{fontWeight: 600}}>{v.title}</div>
                              <div style={{color: 'var(--text-muted)'}}>{parseInt(v.views).toLocaleString()} views</div>
                              {v.predicted_viral === 1 && <div style={{color: 'gold', fontSize: '0.65rem', marginTop: '3px'}}>✦ Predicted Viral</div>}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div style={{color: 'var(--text-muted)', fontSize: '0.85rem'}}>No content detected in latest snapshots.</div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* TAB 3: LIVE PREDICTOR */}
            {predHubTab === 'live' && (
              <div className="animate-fade-in" style={{maxWidth: '800px', margin: '0 auto'}}>
                <div className="card">
                  <div className="card-title"><Zap size={20} /> Part 3: Lightning Predictor (Real XGBoost Inference)</div>
                            <div className="pred-form-group">
                    <label className="pred-form-label">Dán link YouTube Shorts của bạn</label>
                    <div style={{display: 'flex', gap: '10px'}}>
                      <input 
                        type="text" className="search-input" style={{flex: 1, marginBottom: 0}}
                        placeholder="https://www.youtube.com/shorts/..." 
                        value={predictionUrl} onChange={(e) => setPredictionUrl(e.target.value)} 
                      />
                      <button 
                        className="sub-tab active" style={{padding: '0 20px'}}
                        onClick={() => {
                          const match = predictionUrl.match(/[\/\=]([a-zA-Z0-9_-]{11})/);
                          if(match) fetchRealMetadata(match[1]);
                          else alert("Link không đúng định dạng!");
                        }}
                      >Phân tích ngay</button>
                    </div>
                  </div>

                  {metadataError && (
                    <div className="animate-fade-in" style={{marginTop: '20px', padding: '15px', background: 'rgba(239, 68, 68, 0.1)', color: '#ef4444', borderRadius: '8px', border: '1px solid rgba(239, 68, 68, 0.2)'}}>
                      ⚠️ {metadataError}. Vui lòng kiểm tra lại URL hoặc trạng thái Backend.
                    </div>
                  )}

                  {showManualForm && !isPredicting && (
                     <div className="animate-fade-in" style={{marginTop: '20px', padding: '20px', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid var(--border)'}}>
                        <div style={{display: 'flex', gap: '20px', alignItems: 'center', marginBottom: '20px'}}>
                           <img src={liveThumbnail} style={{width: '120px', borderRadius: '8px', boxShadow: '0 4px 8px rgba(0,0,0,0.3)'}} />
                           <div>
                              <div style={{color: 'var(--primary)', fontSize: '0.8rem', fontWeight: 600}}>Đã trích xuất dữ liệu thật từ YouTube API</div>
                              <h4 style={{margin: '5px 0'}}>{liveTitle}</h4>
                              <div style={{fontSize: '0.85rem'}}>Channel: <span style={{color: 'var(--primary)'}}>{liveChannelID}</span></div>
                              <div style={{fontSize: '0.75rem', color: 'var(--text-muted)'}}>Sức mạnh kênh: {Math.round(liveChannelAvg).toLocaleString()} views/video</div>
                           </div>
                        </div>
                        
                        <div className="grid-2">
                          <div className="pred-form-group">
                            <label className="pred-form-label">Lượt xem hiện tại (Real-time)</label>
                            <input type="number" className="search-input" value={liveViews} readOnly />
                          </div>
                          <div className="pred-form-group">
                            <label className="pred-form-label">Thời gian đăng (Published At)</label>
                            <input type="text" className="search-input" value={livePubDate} readOnly />
                          </div>
                        </div>
                        
                        <p style={{fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic', marginTop: '10px'}}>
                           * Model XGBoost đang tính toán view tăng thêm dựa trên momentum {Math.round(liveViews/6) || 0} views/h.
                        </p>
                     </div>
                  )}

                  {isPredicting && (
                    <div style={{textAlign: 'center', padding: '40px'}} className="animate-fade-in">
                       <div className="spinner"></div>
                       <p style={{color: 'var(--text-muted)', marginTop: '15px'}}>Running XGBoost Regressor & Classifier...</p>
                    </div>
                  )}

                  {quickPredResult && !isPredicting && (
                    <div className="animate-fade-in" style={{marginTop: '20px', padding: '25px', background: 'rgba(155, 135, 245, 0.1)', borderRadius: '16px', border: '1px solid var(--primary)'}}>
                       <div style={{fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '5px'}}>{quickPredResult.model}</div>
                       <h3 style={{margin: 0, marginBottom: '20px'}}>{quickPredResult.title}</h3>
                       <div className="grid-3">
                          <div>
                            <div style={{fontSize: '0.75rem', color: 'var(--text-muted)'}}>Dự báo sau 24h</div>
                            <div style={{fontSize: '1.4rem', fontWeight: 700, color: 'var(--primary)'}}>{quickPredResult.views24.toLocaleString()} views</div>
                          </div>
                          <div>
                            <div style={{fontSize: '0.75rem', color: 'var(--text-muted)'}}>Dự báo sau 48h</div>
                            <div style={{fontSize: '1.4rem', fontWeight: 700, color: 'var(--primary)'}}>{quickPredResult.views48.toLocaleString()} views</div>
                          </div>
                          <div>
                            <div style={{fontSize: '0.75rem', color: 'var(--text-muted)'}}>Trạng thái Viral</div>
                            <div className="status-badge" style={{
                               background: quickPredResult.viralStatus === 'VIRAL' ? 'gold' : quickPredResult.viralStatus === 'POTENTIAL' ? 'rgba(74, 222, 128, 0.2)' : 'rgba(255,255,255,0.1)',
                               color: quickPredResult.viralStatus === 'VIRAL' ? 'black' : 'white',
                               fontSize: '1rem', padding: '5px 15px', marginTop: '5px'
                            }}>
                               {quickPredResult.viralStatus}
                            </div>
                          </div>
                       </div>
                       <div style={{marginTop: '20px', padding: '10px', background: 'rgba(0,0,0,0.3)', borderRadius: '8px', fontSize: '0.8rem'}}>
                          ⚖️ <strong>Ghi chú:</strong> Video đang được đăng tải được {quickPredResult.age}h.
                       </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* GROWTH TRACKER TAB */}
        {activeTab === 'growth' && (
          <div className="animate-fade-in">
            {!selectedVideo ? (
              <div className="card" style={{textAlign: 'center', padding: '4rem'}}>
                <Search size={48} color="var(--text-muted)" style={{marginBottom: '1rem'}} />
                <h3 style={{color: 'var(--text-muted)'}}>Select a video from Rankings to view its growth</h3>
                <button 
                  style={{
                    background: 'var(--primary)', color: 'white', border: 'none', 
                    padding: '10px 20px', borderRadius: '6px', cursor: 'pointer', marginTop: '1rem'
                  }}
                  onClick={() => setActiveTab('rankings')}
                >
                  Go to Rankings
                </button>
              </div>
            ) : (
              <div>
                <button 
                  style={{background: 'transparent', color: 'var(--text-muted)', border: '1px solid var(--border)', padding: '6px 12px', borderRadius: '6px', cursor: 'pointer', marginBottom: '1rem'}}
                  onClick={() => setSelectedVideo(null)}
                >
                  &larr; Back
                </button>
                <div className="card">
                  <div className="card-title">
                    Growth Analysis: <span className="text-trunc" style={{maxWidth: '400px', display: 'inline-block'}}>{selectedVideo.title}</span>
                  </div>
                  <div style={{display: 'flex', gap: '2rem', marginBottom: '2rem'}}>
                    <div>
                      <div className="stat-label">Total Views</div>
                      <div className="stat-value" style={{fontSize: '1.5rem'}}>{selectedVideo.views.toLocaleString()}</div>
                    </div>
                    <div>
                      <div className="stat-label">Published At</div>
                      <div className="stat-value" style={{fontSize: '1.5rem', color: 'var(--text-light)'}}>{new Date(selectedVideo.published_at).toLocaleString()}</div>
                    </div>
                  </div>
                  
                  {timeSeriesData === null ? (
                    <div>Loading timeseries data...</div>
                  ) : timeSeriesData.length <= 1 ? (
                    <div style={{color: 'var(--text-muted)'}}>Not enough snapshots available to draw a growth chart (Only {timeSeriesData.length} snapshot found).</div>
                  ) : (
                    <div style={{width: '100%', height: 400}}>
                      <ResponsiveContainer>
                        <LineChart data={timeSeriesData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                          <XAxis dataKey="timeLabel" stroke="var(--text-muted)" />
                          <YAxis stroke="var(--text-muted)" domain={['auto', 'auto']} />
                          <Tooltip 
                            contentStyle={{backgroundColor: 'var(--bg-dark)', borderColor: 'var(--border)'}}
                            itemStyle={{color: 'var(--primary)'}}
                            labelStyle={{color: 'var(--text-light)'}}
                          />
                          <Legend />
                          <Line type="monotone" dataKey="views" name="View Count" stroke="var(--primary)" strokeWidth={3} dot={{r: 4}} activeDot={{r: 8}} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  );
}

export default App;
